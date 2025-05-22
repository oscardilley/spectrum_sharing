""" testing_new_sionna.py

Sandbox for testing out the features and limitations of Sionna v1. Learn here before refactoring the repo. """

import sionna
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Sionna components
from sionna.sys.utils import spread_across_subcarriers
from sionna.sys import PHYAbstraction, \
    OuterLoopLinkAdaptation, gen_hexgrid_topology, \
    get_pathloss, open_loop_uplink_power_control, downlink_fair_power_control, \
    get_num_hex_in_grid, PFSchedulerSUMIMO
from sionna.phy.constants import BOLTZMANN_CONSTANT
from sionna.phy.utils import db_to_lin, dbm_to_watt, log2, insert_dims
from sionna.phy import config, dtypes, Block
from sionna.phy.channel.tr38901 import UMi, UMa, RMa, PanelArray
from sionna.phy.channel import GenerateOFDMChannel
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid, RZFPrecodedChannel, EyePrecodedChannel, \
    LMMSEPostEqualizationSINR

# Channel evolution - REPLACE IN HERE WITH RAY TRACING 
# ---------------------------------------------------------------------------------------------
class ChannelMatrix(Block):
    def __init__(self,
                 resource_grid,
                 batch_size,
                 num_rx,
                 num_tx,
                 coherence_time,
                 precision=None):
        super().__init__(precision=precision)
        self.resource_grid = resource_grid
        self.coherence_time = coherence_time
        self.batch_size = batch_size
        # Fading autoregressive coefficient initialization
        self.rho_fading = config.tf_rng.uniform([batch_size, num_rx, num_tx],
                                                minval=.95,
                                                maxval=.99,
                                                dtype=self.rdtype)
        # Fading initialization
        self.fading = tf.ones([batch_size, num_rx, num_tx],
                              dtype=self.rdtype)

    def call(self, channel_model):
        """ Generate OFDM channel matrix"""

        # Instantiate the OFDM channel generator
        ofdm_channel = GenerateOFDMChannel(channel_model,
                                           self.resource_grid)

        # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]
        h_freq = ofdm_channel(self.batch_size)
        return h_freq

    def update(self,
               channel_model,
               h_freq,
               slot):
        """ Update channel matrix every coherence_time slots """
        # Generate new channel realization
        h_freq_new = self.call(channel_model)

        # Change to new channel every coherence_time slots
        change = tf.cast(tf.math.mod(
            slot, self.coherence_time) == 0, self.cdtype)
        h_freq = change * h_freq_new + \
            (tf.cast(1, self.cdtype) - change) * h_freq
        return h_freq

    def apply_fading(self,
                     h_freq):
        """ Apply fading, modeled as an autoregressive process, to channel matrix """
        # Multiplicative fading factor evolving via an AR process
        # [batch_size, num_rx, num_tx]
        self.fading = tf.cast(1, self.rdtype) - self.rho_fading + self.rho_fading * self.fading + \
            config.tf_rng.uniform(
                self.fading.shape, minval=-.1, maxval=.1, dtype=self.rdtype)
        self.fading = tf.maximum(self.fading, tf.cast(0, self.rdtype))
        # [batch_size, num_rx, 1, num_tx, 1, 1, 1]
        fading_expand = insert_dims(self.fading, 1, axis=2)
        fading_expand = insert_dims(fading_expand, 3, axis=4)

        # Channel matrix in the current slot
        h_freq_fading = tf.cast(tf.math.sqrt(
            fading_expand), self.cdtype) * h_freq
        return h_freq_fading
    

# Configuring how base stations serve users - simple here that nearest serves - WE CAN CHANGE THIS FOR DUAL CONNECTIVITY
# ---------------------------------------------------------------------------------------------
def get_stream_management(direction,
                          num_rx,
                          num_tx,
                          num_streams_per_ut,
                          num_ut_per_sector):
    """
    Instantiate a StreamManagement object.
    It determines which data streams are intended for each receiver
    """
    if direction == 'downlink':
        num_streams_per_tx = num_streams_per_ut * num_ut_per_sector
        # RX-TX association matrix
        rx_tx_association = np.zeros([num_rx, num_tx])
        idx = np.array([[i1, i2] for i2 in range(num_tx) for i1 in
                        np.arange(i2*num_ut_per_sector,
                                  (i2+1)*num_ut_per_sector)])
        rx_tx_association[idx[:, 0], idx[:, 1]] = 1

    else:
        num_streams_per_tx = num_streams_per_ut
        # RX-TX association matrix
        rx_tx_association = np.zeros([num_rx, num_tx])
        idx = np.array([[i1, i2] for i1 in range(num_rx) for i2 in
                        np.arange(i1*num_ut_per_sector,
                                  (i1+1)*num_ut_per_sector)])
        rx_tx_association[idx[:, 0], idx[:, 1]] = 1

    stream_management = StreamManagement(
        rx_tx_association, num_streams_per_tx)
    return stream_management

# Used to get the SINR
# ---------------------------------------------------------------------------------------------
def get_sinr(tx_power,
             stream_management,
             no,
             direction,
             h_freq_fading,
             num_bs,
             num_ut_per_sector,
             num_streams_per_ut,
             resource_grid):
    """ Compute post-equalization SINR. It is assumed:
     - DL: Regularized zero-forcing precoding
     - UL: No precoding, only power allocation
    LMMSE equalizer is used in both DL and UL.
    """
    # tx_power: [batch_size, num_bs, num_tx_per_sector,
    #            num_streams_per_tx, num_ofdm_sym, num_subcarriers]
    # Flatten across sectors
    # [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, num_subcarriers]
    s = tx_power.shape
    tx_power = tf.reshape(tx_power, [s[0], s[1]*s[2]] + s[3:])

    # Compute SINR
    # [batch_size, num_ofdm_sym, num_subcarriers, num_ut,
    #  num_streams_per_ut]
    if direction == 'downlink':
        # Regularized zero-forcing precoding in the DL
        precoded_channel = RZFPrecodedChannel(resource_grid=resource_grid,
                                              stream_management=stream_management)
        h_eff = precoded_channel(h_freq_fading,
                                 tx_power=tx_power,
                                 alpha=no)  # Regularizer
    else:
        # No precoding in the UL: just power allocation
        precoded_channel = EyePrecodedChannel(resource_grid=resource_grid,
                                              stream_management=stream_management)
        h_eff = precoded_channel(h_freq_fading,
                                 tx_power=tx_power)

    # LMMSE equalizer
    lmmse_posteq_sinr = LMMSEPostEqualizationSINR(resource_grid=resource_grid,
                                                  stream_management=stream_management)
    # Post-equalization SINR
    # [batch_size, num_ofdm_symbols, num_subcarriers, num_rx, num_streams_per_rx]
    sinr = lmmse_posteq_sinr(h_eff, no=no, interference_whitening=True)

    # [batch_size, num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut]
    sinr = tf.reshape(
        sinr, sinr.shape[:-2] + [num_bs*num_ut_per_sector, num_streams_per_ut])

    # Regroup by sector
    # [batch_size, num_ofdm_symbols, num_subcarriers, num_bs, num_ut_per_sector, num_streams_per_ut]
    sinr = tf.reshape(
        sinr, sinr.shape[:-2] + [num_bs, num_ut_per_sector, num_streams_per_ut])

    # [batch_size, num_bs, num_ofdm_sym, num_subcarriers, num_ut_per_sector, num_streams_per_ut]
    sinr = tf.transpose(sinr, [0, 3, 1, 2, 4, 5])
    return sinr


# Used for scheduling
def estimate_achievable_rate(sinr_eff_db_last,
                             num_ofdm_sym,
                             num_subcarriers):
    """ Estimate achievable rate """
    # [batch_size, num_bs, num_ut_per_sector]
    rate_achievable_est = log2(tf.cast(1, sinr_eff_db_last.dtype) +
                               db_to_lin(sinr_eff_db_last))

    # Broadcast to time/frequency grid
    # [batch_size, num_bs, num_ofdm_sym, num_subcarriers, num_ut_per_sector]
    rate_achievable_est = insert_dims(
        rate_achievable_est, 2, axis=-2)
    rate_achievable_est = tf.tile(rate_achievable_est,
                                  [1, 1, num_ofdm_sym, num_subcarriers, 1])
    return rate_achievable_est


# Functions for recording metrics
# ---------------------------------------------------------------------------------------------
def init_result_history(batch_size,
                        num_slots,
                        num_bs,
                        num_ut_per_sector):
    """ Initialize dictionary containing history of results """
    hist = {}
    for key in ['pathloss_serving_cell',
                'tx_power', 'olla_offset',
                'sinr_eff', 'pf_metric',
                'num_decoded_bits', 'mcs_index',
                'harq', 'num_allocated_re']:
        hist[key] = tf.TensorArray(
            size=num_slots,
            element_shape=[batch_size,
                           num_bs,
                           num_ut_per_sector],
            dtype=tf.float32)
    return hist


def record_results(hist,
                   slot,
                   sim_failed=False,
                   pathloss_serving_cell=None,
                   num_allocated_re=None,
                   tx_power_per_ut=None,
                   num_decoded_bits=None,
                   mcs_index=None,
                   harq_feedback=None,
                   olla_offset=None,
                   sinr_eff=None,
                   pf_metric=None,
                   shape=None):
    """ Record results of last slot """
    if not sim_failed:
        for key, value in zip(['pathloss_serving_cell', 'olla_offset', 'sinr_eff',
                               'num_allocated_re', 'tx_power', 'num_decoded_bits',
                               'mcs_index', 'harq'],
                              [pathloss_serving_cell, olla_offset, sinr_eff,
                               num_allocated_re, tx_power_per_ut, num_decoded_bits,
                               mcs_index, harq_feedback]):
            hist[key] = hist[key].write(slot, tf.cast(value, tf.float32))
        # Average PF metric across resources
        hist['pf_metric'] = hist['pf_metric'].write(
            slot, tf.reduce_mean(pf_metric, axis=[-2, -3]))
    else:
        nan_tensor = tf.cast(tf.fill(shape,
                                     float('nan')), dtype=tf.float32)
        for key in hist:
            hist[key] = hist[key].write(slot, nan_tensor)
    return hist


def clean_hist(hist, batch=0):
    """ Extract batch, convert to Numpy, and mask metrics when user is not
    scheduled """
    # Extract batch and convert to Numpy
    for key in hist:
        try:
            # [num_slots, num_bs, num_ut_per_sector]
            hist[key] = hist[key].numpy()[:, batch, :, :]
        except:
            pass

    # Mask metrics when user is not scheduled
    hist['mcs_index'] = np.where(
        hist['harq'] == -1, np.nan, hist['mcs_index'])
    hist['sinr_eff'] = np.where(
        hist['harq'] == -1, np.nan, hist['sinr_eff'])
    hist['tx_power'] = np.where(
        hist['harq'] == -1, np.nan, hist['tx_power'])
    hist['num_allocated_re'] = np.where(
        hist['harq'] == -1, 0, hist['num_allocated_re'])
    hist['harq'] = np.where(
        hist['harq'] == -1, np.nan, hist['harq'])
    return hist

# System level block with XLA compilation for scaling
# ---------------------------------------------------------------------------------------------
class SystemLevelSimulator(Block):
    def __init__(self,
                 batch_size,
                 num_rings,
                 num_ut_per_sector,
                 carrier_frequency,
                 resource_grid,
                 scenario,
                 direction,
                 ut_array,
                 bs_array,
                 bs_max_power_dbm,
                 ut_max_power_dbm,
                 coherence_time,
                 pf_beta=0.98,
                 max_bs_ut_dist=None,
                 min_bs_ut_dist=None,
                 temperature=294,
                 o2i_model='low',
                 average_street_width=20.0,
                 average_building_height=5.0,
                 precision=None):
        super().__init__(precision=precision)

        assert scenario in ['umi', 'uma', 'rma']
        assert direction in ['uplink', 'downlink']
        self.scenario = scenario
        self.batch_size = int(batch_size)
        self.resource_grid = resource_grid
        self.num_ut_per_sector = int(num_ut_per_sector)
        self.direction = direction
        self.bs_max_power_dbm = bs_max_power_dbm  # [dBm]
        self.ut_max_power_dbm = ut_max_power_dbm  # [dBm]
        self.coherence_time = tf.cast(coherence_time, tf.int32)  # [slots]
        num_cells = get_num_hex_in_grid(num_rings)
        self.num_bs = num_cells * 3
        self.num_ut = self.num_bs * self.num_ut_per_sector
        self.num_ut_ant = ut_array.num_ant
        self.num_bs_ant = bs_array.num_ant
        if bs_array.polarization == 'dual':
            self.num_bs_ant *= 2
        if self.direction == 'uplink':
            self.num_tx, self.num_rx = self.num_ut, self.num_bs
            self.num_tx_ant, self.num_rx_ant = self.num_ut_ant, self.num_bs_ant
            self.num_tx_per_sector = self.num_ut_per_sector
        else:
            self.num_tx, self.num_rx = self.num_bs, self.num_ut
            self.num_tx_ant, self.num_rx_ant = self.num_bs_ant, self.num_ut_ant
            self.num_tx_per_sector = 1

        # Assume 1 stream for UT antenna
        self.num_streams_per_ut = resource_grid.num_streams_per_tx

        # Set TX-RX pairs via StreamManagement
        self.stream_management = get_stream_management(direction,
                                                       self.num_rx,
                                                       self.num_tx,
                                                       self.num_streams_per_ut,
                                                       num_ut_per_sector)
        # Noise power per subcarrier
        self.no = tf.cast(BOLTZMANN_CONSTANT * temperature *
                          resource_grid.subcarrier_spacing, self.rdtype)

        # Slot duration [sec]
        self.slot_duration = resource_grid.ofdm_symbol_duration * \
            resource_grid.num_ofdm_symbols

        # Initialize channel model based on scenario
        self._setup_channel_model(
            scenario, carrier_frequency, o2i_model, ut_array, bs_array,
            average_street_width, average_building_height)

        # Generate multicell topology
        self._setup_topology(num_rings, min_bs_ut_dist, max_bs_ut_dist)

        # Instantiate a PHY abstraction object
        self.phy_abs = PHYAbstraction(precision=self.precision)

        # Instantiate a link adaptation object
        self.olla = OuterLoopLinkAdaptation(
            self.phy_abs,
            self.num_ut_per_sector,
            batch_size=[self.batch_size, self.num_bs])

        # Instantiate a scheduler object
        self.scheduler = PFSchedulerSUMIMO(
            self.num_ut_per_sector,
            resource_grid.fft_size,
            resource_grid.num_ofdm_symbols,
            batch_size=[self.batch_size, self.num_bs],
            num_streams_per_ut=self.num_streams_per_ut,
            beta=pf_beta,
            precision=self.precision)

    def _setup_channel_model(self, scenario, carrier_frequency, o2i_model,
                             ut_array, bs_array, average_street_width,
                             average_building_height):
        """ Initialize appropriate channel model based on scenario """
        common_params = {
            'carrier_frequency': carrier_frequency,
            'ut_array': ut_array,
            'bs_array': bs_array,
            'direction': self.direction,
            'enable_pathloss': True,
            'enable_shadow_fading': True,
            'precision': self.precision
        }

        if scenario == 'umi':  # Urban micro-cell
            self.channel_model = UMi(o2i_model=o2i_model, **common_params)
        elif scenario == 'uma':  # Urban macro-cell
            self.channel_model = UMa(o2i_model=o2i_model, **common_params)
        elif scenario == 'rma':  # Rural macro-cell
            self.channel_model = RMa(
                average_street_width=average_street_width,
                average_building_height=average_building_height,
                **common_params)

    def _setup_topology(self, num_rings, min_bs_ut_dist, max_bs_ut_dist):
        """G enerate and set up network topology """
        self.ut_loc, self.bs_loc, self.ut_orientations, self.bs_orientations, \
            self.ut_velocities, self.in_state, self.los, self.bs_virtual_loc, self.grid = \
            gen_hexgrid_topology(
                batch_size=self.batch_size,
                num_rings=num_rings,
                num_ut_per_sector=self.num_ut_per_sector,
                min_bs_ut_dist=min_bs_ut_dist,
                max_bs_ut_dist=max_bs_ut_dist,
                scenario=self.scenario,
                los=True,
                return_grid=True,
                precision=self.precision)

        # Set topology in channel model
        self.channel_model.set_topology(
            self.ut_loc, self.bs_loc, self.ut_orientations,
            self.bs_orientations, self.ut_velocities,
            self.in_state, self.los, self.bs_virtual_loc)

    def _reset(self,
               bler_target,
               olla_delta_up):
        """  Reset OLLA and HARQ/SINR feedback """
        # Link Adaptation
        self.olla.reset()
        self.olla.bler_target = bler_target
        self.olla.olla_delta_up = olla_delta_up

        # HARQ feedback (no feedback, -1)
        last_harq_feedback = - tf.ones(
            [self.batch_size, self.num_bs, self.num_ut_per_sector],
            dtype=tf.int32)

        # SINR feedback
        sinr_eff_feedback = tf.ones(
            [self.batch_size, self.num_bs, self.num_ut_per_sector],
            dtype=self.rdtype)

        # N. decoded bits
        num_decoded_bits = tf.zeros(
            [self.batch_size, self.num_bs, self.num_ut_per_sector],
            tf.int32)
        return last_harq_feedback, sinr_eff_feedback, num_decoded_bits

    def _group_by_sector(self,
                         tensor):
        """ Group tensor by sector
        - Input: [batch_size, num_ut, num_ofdm_symbols]
        - Output: [batch_size, num_bs, num_ofdm_symbols, num_ut_per_sector]
        """
        tensor = tf.reshape(tensor, [self.batch_size,
                                     self.num_bs,
                                     self.num_ut_per_sector,
                                     self.resource_grid.num_ofdm_symbols])
        # [batch_size, num_bs, num_ofdm_symbols, num_ut_per_sector]
        return tf.transpose(tensor, [0, 1, 3, 2])

    @tf.function(jit_compile=True)
    def call(self,
             num_slots,
             alpha_ul,
             p0_dbm_ul,
             bler_target,
             olla_delta_up,
             mcs_table_index=1,
             fairness_dl=0,
             guaranteed_power_ratio_dl=0.5):

        # -------------- #
        # Initialization #
        # -------------- #
        # Initialize result history
        hist = init_result_history(self.batch_size,
                                   num_slots,
                                   self.num_bs,
                                   self.num_ut_per_sector)

        # Reset OLLA and HARQ/SINR feedback
        last_harq_feedback, sinr_eff_feedback, num_decoded_bits = \
            self._reset(bler_target, olla_delta_up)

        # Initialize channel matrix
        self.channel_matrix = ChannelMatrix(self.resource_grid,
                                            self.batch_size,
                                            self.num_rx,
                                            self.num_tx,
                                            self.coherence_time,
                                            precision=self.precision)
        # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_sym,
        #  num_subcarriers]
        h_freq = self.channel_matrix(self.channel_model)

        # --------------- #
        # Simulate a slot #
        # --------------- #
        def simulate_slot(slot,
                          hist,
                          harq_feedback,
                          sinr_eff_feedback,
                          num_decoded_bits,
                          h_freq):
            try:
                # ------- #
                # Channel #
                # ------- #
                # Update channel matrix
                h_freq = self.channel_matrix.update(self.channel_model,
                                                    h_freq,
                                                    slot)

                # Apply fading
                h_freq_fading = self.channel_matrix.apply_fading(h_freq)

                # --------- #
                # Scheduler #
                # --------- #
                # Estimate achievable rate
                # [batch_size, num_bs, num_ofdm_sym, num_subcarriers, num_ut_per_sector]
                rate_achievable_est = estimate_achievable_rate(
                    self.olla.sinr_eff_db_last,
                    self.resource_grid.num_ofdm_symbols,
                    self.resource_grid.fft_size)

                # SU-MIMO Proportional Fairness scheduler
                # [batch_size, num_bs, num_ofdm_sym, num_subcarriers,
                #  num_ut_per_sector, num_streams_per_ut]
                is_scheduled = self.scheduler(
                    num_decoded_bits,
                    rate_achievable_est)

                # N. allocated subcarriers
                num_allocated_sc = tf.minimum(tf.reduce_sum(
                    tf.cast(is_scheduled, tf.int32), axis=-1), 1)
                # [batch_size, num_bs, num_ofdm_sym, num_ut_per_sector]
                num_allocated_sc = tf.reduce_sum(
                    num_allocated_sc, axis=-2)

                # N. allocated resources per slot
                # [batch_size, num_bs, num_ut_per_sector]
                num_allocated_re = \
                    tf.reduce_sum(tf.cast(is_scheduled, tf.int32),
                                  axis=[-1, -3, -4])

                # ------------- #
                # Power control #
                # ------------- #
                # Compute pathloss
                # [batch_size, num_rx, num_tx, num_ofdm_symbols], [batch_size, num_ut, num_ofdm_symbols]
                pathloss_all_pairs, pathloss_serving_cell = get_pathloss(
                    h_freq_fading,
                    rx_tx_association=tf.convert_to_tensor(
                        self.stream_management.rx_tx_association))
                # Group by sector
                # [batch_size, num_bs, num_ofdm_symbols, num_ut_per_sector]
                pathloss_serving_cell = self._group_by_sector(
                    pathloss_serving_cell)

                if self.direction == 'uplink':
                    # Open-loop uplink power control
                    # [batch_size, num_bs, num_ofdm_symbols, num_ut_per_sector]
                    tx_power_per_ut = open_loop_uplink_power_control(
                        pathloss_serving_cell,
                        num_allocated_sc,
                        alpha=alpha_ul,
                        p0_dbm=p0_dbm_ul,
                        ut_max_power_dbm=self.ut_max_power_dbm)
                else:
                    # Channel quality estimation:
                    # Estimate interference from neighboring base stations
                    # [batch_size, num_ut, num_ofdm_symbols]

                    one = tf.cast(1, pathloss_serving_cell.dtype)

                    # Total received power
                    # [batch_size, num_ut, num_ofdm_symbols]
                    rx_power_tot = tf.reduce_sum(
                        one / pathloss_all_pairs, axis=-2)
                    # [batch_size, num_bs, num_ut_per_sector, num_ofdm_symbols]
                    rx_power_tot = self._group_by_sector(rx_power_tot)

                    # Interference from neighboring base stations
                    interference_dl = rx_power_tot - one / pathloss_serving_cell
                    interference_dl *= dbm_to_watt(self.bs_max_power_dbm)

                    # Fair downlink power allocation
                    # [batch_size, num_bs, num_ofdm_symbols, num_ut_per_sector]
                    tx_power_per_ut, _ = downlink_fair_power_control(
                        pathloss_serving_cell,
                        interference_dl + self.no,
                        num_allocated_sc,
                        bs_max_power_dbm=self.bs_max_power_dbm,
                        guaranteed_power_ratio=guaranteed_power_ratio_dl,
                        fairness=fairness_dl,
                        precision=self.precision)

                # For each user, distribute the power uniformly across
                # subcarriers and streams
                # [batch_size, num_bs, num_tx_per_sector,
                #  num_streams_per_tx, num_ofdm_sym, num_subcarriers]
                tx_power = spread_across_subcarriers(
                    tx_power_per_ut,
                    is_scheduled,
                    num_tx=self.num_tx_per_sector,
                    precision=self.precision)

                # --------------- #
                # Per-stream SINR #
                # --------------- #
                # [batch_size, num_bs, num_ofdm_sym, num_subcarriers,
                #  num_ut_per_sector, num_streams_per_ut]
                sinr = get_sinr(tx_power,
                                self.stream_management,
                                self.no,
                                self.direction,
                                h_freq_fading,
                                self.num_bs,
                                self.num_ut_per_sector,
                                self.num_streams_per_ut,
                                self.resource_grid)

                # --------------- #
                # Link adaptation #
                # --------------- #
                # [batch_size, num_bs, num_ut_per_sector]
                mcs_index = self.olla(num_allocated_re,
                                      harq_feedback=harq_feedback,
                                      sinr_eff=sinr_eff_feedback)

                # --------------- #
                # PHY abstraction #
                # --------------- #
                # [batch_size, num_bs, num_ut_per_sector]
                num_decoded_bits, harq_feedback, sinr_eff, _, _ = self.phy_abs(
                    mcs_index,
                    sinr=sinr,
                    mcs_table_index=mcs_table_index,
                    mcs_category=int(self.direction == 'downlink'))

                # ------------- #
                # SINR feedback #
                # ------------- #
                # [batch_size, num_bs, num_ut_per_sector]
                sinr_eff_feedback = tf.where(num_allocated_re > 0,
                                             sinr_eff,
                                             tf.cast(0., self.rdtype))

                # Record results
                hist = record_results(hist,
                                      slot,
                                      sim_failed=False,
                                      pathloss_serving_cell=tf.reduce_sum(
                                          pathloss_serving_cell, axis=-2),
                                      num_allocated_re=num_allocated_re,
                                      tx_power_per_ut=tf.reduce_sum(
                                          tx_power_per_ut, axis=-2),
                                      num_decoded_bits=num_decoded_bits,
                                      mcs_index=mcs_index,
                                      harq_feedback=harq_feedback,
                                      olla_offset=self.olla.offset,
                                      sinr_eff=sinr_eff,
                                      pf_metric=self.scheduler.pf_metric)

            except tf.errors.InvalidArgumentError as e:
                print(f"SINR computation did not succeed at slot {slot}.\n"
                      f"Error message: {e}. Skipping slot...")
                hist = record_results(hist, slot,
                                      shape=[self.batch_size,
                                             self.num_bs,
                                             self.num_ut_per_sector], sim_failed=True)

            # ------------- #
            # User mobility #
            # ------------- #
            self.ut_loc = self.ut_loc + self.ut_velocities * self.slot_duration

            # Set topology in channel model
            self.channel_model.set_topology(
                self.ut_loc, self.bs_loc, self.ut_orientations,
                self.bs_orientations, self.ut_velocities,
                self.in_state, self.los, self.bs_virtual_loc)

            return [slot + 1, hist, harq_feedback, sinr_eff_feedback,
                    num_decoded_bits, h_freq]

        # --------------- #
        # Simulation loop #
        # --------------- #
        _, hist, *_ = tf.while_loop(
            lambda i, *_: i < num_slots,
            simulate_slot,
            [0, hist, last_harq_feedback, sinr_eff_feedback,
             num_decoded_bits, h_freq])

        for key in hist:
            hist[key] = hist[key].stack()
        return hist
    

# Communication direction
direction = 'downlink'  # 'uplink' or 'downlink'

# 3GPP scenario parameters
scenario = 'umi'  # 'umi', 'uma' or 'rma'

# Number of rings of the hexagonal grid
# With num_rings=1, 7*3=21 base stations are placed
num_rings = 1

# N. users per sector
num_ut_per_sector = 10

# Max/min distance between base station and served users
max_bs_ut_dist = 80  # [m]
min_bs_ut_dist = 0  # [m]

# Carrier frequency
carrier_frequency = 3.5e9  # [Hz]

# Transmit power for base station and user terminals
bs_max_power_dbm = 56  # [dBm]
ut_max_power_dbm = 26  # [dBm]

# Channel is regenerated every coherence_time slots
coherence_time = 100  # [slots]

# MCS table index
# Ranges within [1;4] for downlink and [1;2] for uplink, as in TS 38.214
mcs_table_index = 1

# Number of examples
batch_size = 1

# Create the antenna arrays at the base stations
bs_array = PanelArray(num_rows_per_panel=2,
                      num_cols_per_panel=3,
                      polarization='dual',
                      polarization_type='VH',
                      antenna_pattern='38.901',
                      carrier_frequency=carrier_frequency)

# Create the antenna array at the user terminals
ut_array = PanelArray(num_rows_per_panel=1,
                      num_cols_per_panel=1,
                      polarization='single',
                      polarization_type='V',
                      antenna_pattern='omni',
                      carrier_frequency=carrier_frequency)

# n. OFDM symbols, i.e., time samples, in a slot
num_ofdm_sym = 1
# N. available subcarriers
num_subcarriers = 128
# Subcarrier spacing, i.e., bandwitdh width of each subcarrier
subcarrier_spacing = 15e3  # [Hz]

# Create the OFDM resource grid
resource_grid = ResourceGrid(num_ofdm_symbols=num_ofdm_sym,
                             fft_size=num_subcarriers,
                             subcarrier_spacing=subcarrier_spacing,
                             num_tx=num_ut_per_sector,
                             num_streams_per_tx=ut_array.num_ant)

# Initialize SYS object
sls = SystemLevelSimulator(
    batch_size,
    num_rings,
    num_ut_per_sector,
    carrier_frequency,
    resource_grid,
    scenario,
    direction,
    ut_array,
    bs_array,
    bs_max_power_dbm,
    ut_max_power_dbm,
    coherence_time,
    max_bs_ut_dist=max_bs_ut_dist,
    min_bs_ut_dist=min_bs_ut_dist,
    temperature=294,  # Environment temperature for noise power computation
    o2i_model='low',  # 'low' or 'high',
    average_street_width=20.,
    average_building_height=10.)

fig = sls.grid.show()
ax = fig.get_axes()
ax[0].plot(sls.ut_loc[0, :, 0], sls.ut_loc[0, :, 1],
           'xk', label='user position')
ax[0].legend()
plt.savefig('hex_grid.png')