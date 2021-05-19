import math
import numpy as np
import scipy.signal as ss
from scipy.io import loadmat
import wfdb
from ..denoise import wt_denoise, median_denoise, normalize
from ..detection import detect_qrs, detect_beats, qrs_peak_detect
from ..ecg_plot import plot_12, save_as_png


class ECGRecord(object):

    def __init__(self, path, format):
        self.path = path
        self.format = format
        self.lead_match = None
        self.fs = None
        self.raw_data = None
        self.symbol = None  # stay list
        self.coords = None  # stay list
        self.diagnose = None
        self.fixed_len = 200
        self.label_name = None
        self.beat_label = None
        if format == 'qt' or format == 'qtdb':
            self._load_qt(path)
        elif format == 'ludb':
            self._load_ludb(path)
        elif format == 'unlabeled':
            self._load_unlabeled(path)
        elif format == 'mat' or format == 'matlab':
            self._load_mat(path)
        else:
            print("Invalid ECG data format.")
            raise NotImplementedError
        assert isinstance(self.symbol, list)
        assert isinstance(self.lead_match, list)
        self.lead_dict = dict()
        for i in range(len(self.lead_match)):
            self.lead_dict[self.lead_match[i]] = i

    def _load_mat(self, path):
        mat = loadmat(path)
        self.fs = 500
        leads_name = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.lead_match = []
        raw_data = []
        for lead in leads_name:
            if lead in mat.keys():
                self.lead_match.append(lead)
                raw_data.append([mat[lead]])
        self.raw_data = np.concatenate(raw_data, axis=0)
        self.symbol = []
        self.coords = []
        self.label_name = None
        self._generate_beatlabel_from_estimation()

    def _load_qt(self, path):
        """
        2 leads, NO diagnose
        :param path:
        :return:
        """
        signal, info = wfdb.rdsamp(path)
        ann_1 = wfdb.rdann(path, extension='q1c')
        ann_2 = wfdb.rdann(path, extension='qt1')
        self.fs = 250
        self.lead_match = ['anonymous1', 'anonymous2']
        self.raw_data = np.transpose(np.array([signal]), (2, 0, 1))
        self.symbol = [ann_1.symbol, ann_2.symbol]
        self.coords = [ann_1.sample, ann_2.sample]
        if list(np.unique(np.array(self.symbol[0]))) != ['(', ')', 'N', 'p', 't'] and list(np.unique(np.array(self.symbol[0]))) != ['(', ')', 'N', 'p', 't', 'u']:
            print("Invalid symbols in ECG annotations.")
            raise ValueError
        self.label_name = ['(', 'p', ')', '(', 'N', ')', 't', ')']
        self._generate_beatlabel_from_label()

    def _load_ludb(self, path):
        """
        12 leads, NO diagnose
        :param path:
        :return:
        """
        signal, info = wfdb.rdsamp(path)
        self.fs = 500
        self.lead_match = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.raw_data = np.transpose(np.array([signal]), (2, 0, 1))
        self.symbol = []
        self.coords = []
        for lead in ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']:
            ann_ii = wfdb.rdann(path, extension='atr_{}'.format(lead))
            symbol_1 = ann_ii.symbol
            coords_1 = ann_ii.sample
            if list(np.unique(np.array(symbol_1))) != ['(', ')', 'N', 'p', 't'] and list(np.unique(np.array(symbol_1))) != ['(', ')', 'N', 'p', 't', 'u']:
                print("Invalid symbols in ECG annotations.")
                raise ValueError
            self.symbol.append(symbol_1)
            self.coords.append(coords_1)
        self.label_name = ['(', 'p', ')', '(', 'N', ')', '(', 't', ')']
        self._generate_beatlabel_from_label()

    def _load_unlabeled(self, path):
        """
        2 lead_match, NO diagnose, NO symbol
        :param path:
        :return:
        """
        signal, info = wfdb.rdsamp(path)
        self.fs = 250
        self.lead_match = ['anonymous1', 'anonymous2']
        self.raw_data = np.transpose(np.array([signal]), (2, 0, 1))
        self.symbol = []
        self.coords = []
        self.label_name = None
        self._generate_beatlabel_from_estimation()

    def _generate_beatlabel_from_label(self):
        if self.format == 'qt' or self.format == 'qtdb':
            lead = 0
        elif self.format == 'ludb':
            lead = 1
        elif self.format == 'unlabeled' or self.format == 'mat' or self.format == 'matlab':
            print("No label info found in ecg records.")
            raise ValueError
        else:
            print("Invalid ECG data format.")
            raise ValueError
        self.beat_label = []
        symbol = self.symbol[lead]
        coords = self.coords[lead]
        qrs_locations = []
        for k in range(len(symbol)):
            if symbol[k] == 'N':
                qrs_locations.append(coords[k])
        for qrs_ in qrs_locations:
            index = np.argwhere(coords == qrs_)[0][0]
            label_dict = dict()
            if symbol[index - 4: index + 4] == ['(', 'p', ')', '(', 'N', ')', 't', ')']:
                label_dict['Ponset'] = [coords[index-4]]
                label_dict['P'] = [coords[index-3]]
                label_dict['Poffset'] = [coords[index-2]]
                label_dict['Ronset'] = [coords[index-1]]
                label_dict['R'] = [coords[index]]
                label_dict['Roffset'] = [coords[index+1]]
                label_dict['Tonset'] = []
                label_dict['T'] = [coords[index+2]]
                label_dict['Toffset'] = [coords[index+3]]
            elif symbol[index - 4: index + 5] == ['(', 'p', ')', '(', 'N', ')', '(', 't', ')']:
                label_dict['Ponset'] = [coords[index-4]]
                label_dict['P'] = [coords[index-3]]
                label_dict['Poffset'] = [coords[index-2]]
                label_dict['Ronset'] = [coords[index-1]]
                label_dict['R'] = [coords[index]]
                label_dict['Roffset'] = [coords[index+1]]
                label_dict['Tonset'] = [coords[index+2]]
                label_dict['T'] = [coords[index+3]]
                label_dict['Toffset'] = [coords[index+4]]
            else:
                continue
            self.beat_label.append(label_dict)

    def _generate_beatlabel_from_estimation(self):
        # Not implemented
        pass

    def preprocess(self, reference_lead=None, left_rates=None, right_rates=None, fixed_len=None):
        # TODO: optimize in resampling process
        if fixed_len is None:
            fixed_len = self.fixed_len
        if self.format == 'qt' or self.format == 'qtdb':
            return self._preprocess_qt(left_rates=left_rates, right_rates=right_rates, fixed_len=fixed_len)
        elif self.format == 'ludb':
            return self._preprocess_ludb(lead=reference_lead, left_rates=left_rates, right_rates=right_rates, fixed_len=fixed_len)
        elif self.format == 'unlabeled':
            return self._preprocess_unlabeled(fixed_len=fixed_len)
        elif self.format == 'mat' or self.format == 'matlab':
            return self._preprocess_unlabeled(fixed_len=fixed_len)
        else:
            raise ValueError

    def _preprocess_qt(self, left_rates=None, right_rates=None, fixed_len=None):
        if left_rates is None or right_rates is None:
            raise ValueError
        symbol = self.symbol[0]
        coords = self.coords[0]
        x = median_denoise(wt_denoise(self.raw_data[0, 0, :]), fs=self.fs)
        qrs_pos = np.array(detect_qrs(x, fs=self.fs))
        avg_rr = np.mean(qrs_pos[1:] - qrs_pos[:-1])
        qrs_locations = []
        for k in range(len(symbol)):
            if symbol[k] == 'N':
                qrs_locations.append(coords[k])
        segments0 = []
        labels0 = []
        raw_lengths0 = []
        for l in range(len(self.lead_match)):
            segments1 = []
            labels1 = []
            raw_lengths1 = []
            x = median_denoise(wt_denoise(self.raw_data[l, 0, :]), fs=self.fs)
            for lr in left_rates:
                for rr in right_rates:
                    for qrs_ in qrs_locations:
                        index = np.argwhere(coords == qrs_)[0][0]

                        if symbol[index - 4: index + 4] == ['(', 'p', ')', '(', 'N', ')', 't', ')']:
                            start = coords[index - 4]
                            end = coords[index + 3]
                            gap = int(0.1 * len(x[start: end + 1]))
                            left_bound = min(qrs_ - int(avg_rr * lr), start - gap)
                            right_bound = max(qrs_ + int(avg_rr * rr), end + gap)
                            raw_segment = x[left_bound: right_bound]
                            raw_len = len(raw_segment)
                            raw_lengths1.append(raw_len)
                            label = coords[index - 4: index + 4] - left_bound
                            _tmp_segment = ss.resample(raw_segment, raw_len * fixed_len)
                            _tmp_segment = np.reshape(_tmp_segment, newshape=[fixed_len, raw_len])
                            resampled_segment = np.mean(_tmp_segment, axis=1)
                            label = (label / raw_len * fixed_len).astype(np.int32)
                            label = list(label)
                            label = np.array(label)
                            if np.min(label) > 0 and np.max(label) < fixed_len:
                                segments1.append(np.expand_dims(resampled_segment, axis=0))
                                labels1.append(np.expand_dims(label, axis=0))

                        elif symbol[index - 4: index + 5] == ['(', 'p', ')', '(', 'N', ')', '(', 't', ')']:
                            start = coords[index - 4]
                            end = coords[index + 4]
                            gap = int(0.1 * len(x[start: end + 1]))
                            left_bound = min(qrs_ - int(avg_rr * lr), start - gap)
                            right_bound = max(qrs_ + int(avg_rr * rr), end + gap)
                            raw_segment = x[left_bound: right_bound]
                            raw_len = len(raw_segment)
                            raw_lengths1.append(raw_len)
                            label = coords[index - 4: index + 5] - left_bound
                            _tmp_segment = ss.resample(raw_segment, raw_len * fixed_len)
                            _tmp_segment = np.reshape(_tmp_segment, newshape=[fixed_len, raw_len])
                            resampled_segment = np.mean(_tmp_segment, axis=1)
                            label = (label / raw_len * fixed_len).astype(np.int32)
                            label = label[[0, 1, 2, 3, 4, 5, 7, 8]]
                            if np.min(label) > 0 and np.max(label) < fixed_len:
                                segments1.append(np.expand_dims(resampled_segment, axis=0))
                                labels1.append(np.expand_dims(label, axis=0))

                        else:
                            continue

            if len(segments1) > 0:
                segments1 = normalize(np.concatenate(segments1, axis=0))
                labels1 = np.concatenate(labels1, axis=0)
                segments0.append(segments1)
                labels0.append(labels1)
                raw_lengths0.append(np.array(raw_lengths1))

        return np.array(segments0), np.array(labels0), np.array(raw_lengths0), self.diagnose

    def _preprocess_ludb(self, lead=None, left_rates=None, right_rates=None, fixed_len=None):
        if left_rates is None or right_rates is None:
            raise ValueError
        if lead is None:
            lead = 1
        symbol = self.symbol[lead]
        coords = self.coords[lead]
        x = median_denoise(wt_denoise(self.raw_data[lead, 0, :]), fs=self.fs)
        qrs_pos = np.array(detect_qrs(x, fs=self.fs))
        avg_rr = np.mean(qrs_pos[1:] - qrs_pos[:-1])
        qrs_locations = []
        for k in range(len(symbol)):
            if symbol[k] == 'N':
                qrs_locations.append(coords[k])
        segments0 = []
        labels0 = []
        raw_lengths0 = []
        for l in range(len(self.lead_match)):
            segments1 = []
            labels1 = []
            raw_lengths1 = []
            x = median_denoise(wt_denoise(self.raw_data[l, 0, :]), fs=self.fs)
            for lr in left_rates:
                for rr in right_rates:
                    for qrs_ in qrs_locations:
                        index = np.argwhere(coords == qrs_)[0][0]

                        if symbol[index - 4: index + 5] == ['(', 'p', ')', '(', 'N', ')', '(', 't', ')']:
                            start = coords[index - 4]
                            end = coords[index + 4]
                            gap = int(0.1 * len(x[start: end + 1]))
                            left_bound = min(qrs_ - int(avg_rr * lr), start - gap)
                            right_bound = max(qrs_ + int(avg_rr * rr), end + gap)
                            raw_segment = x[left_bound: right_bound]
                            index_array = np.arange(0, len(raw_segment))
                            qrs_s = qrs_ - left_bound
                            a_l = index_array[0: qrs_s]
                            a_r = index_array[qrs_s:]
                            a_l = 1 / (1 + np.exp(-(a_l - max(qrs_s - 250, qrs_s - 0.7 * avg_rr)) / 50))
                            a_r = 1 - 1 / (1 + np.exp(-(a_r - min(qrs_s + 250, qrs_s + 0.5 * avg_rr)) / 50))
                            a_ = np.concatenate([a_l, a_r])
                            raw_segment = np.multiply(raw_segment, a_)
                            raw_len = len(raw_segment)
                            raw_lengths1.append(raw_len)
                            label = coords[index - 4: index + 5] - left_bound
                            _tmp_segment = ss.resample(raw_segment, raw_len * fixed_len)
                            _tmp_segment = np.reshape(_tmp_segment, newshape=[fixed_len, raw_len])
                            resampled_segment = np.mean(_tmp_segment, axis=1)
                            label = (label / raw_len * fixed_len).astype(np.int32)
                            if np.min(label) > 0 and np.max(label) < fixed_len:
                                segments1.append(np.expand_dims(resampled_segment, axis=0))
                                labels1.append(np.expand_dims(label, axis=0))

                        else:
                            continue
            if len(segments1) > 0:
                segments1 = normalize(np.concatenate(segments1, axis=0))
                labels1 = np.concatenate(labels1, axis=0)
                segments0.append(segments1)
                labels0.append(labels1)
                raw_lengths0.append(np.array(raw_lengths1))

        return np.array(segments0), np.array(labels0), np.array(raw_lengths0), self.diagnose

    def _preprocess_unlabeled(self, fixed_len=None):
        x = median_denoise(wt_denoise(self.raw_data[0, 0, :]), fs=self.fs)
        qrs_pos = np.array(detect_qrs(x, fs=self.fs))
        avg_rr = np.mean(qrs_pos[1:] - qrs_pos[:-1])
        segments0 = []
        raw_lengths0 = []
        for l in range(len(self.lead_match)):
            segments1 = []
            raw_lengths1 = []
            x = median_denoise(wt_denoise(self.raw_data[l, 0, :]), fs=self.fs)
            for k in range(1, len(qrs_pos) - 1):
                qrs_ = qrs_pos[k]
                bound_l = qrs_ - int(avg_rr * 0.45)
                bound_r = qrs_ + int(avg_rr * 0.55)
                seg = x[bound_l: bound_r]
                raw_len = len(seg)
                raw_lengths1.append(raw_len)
                resampled_seg = ss.resample(seg, raw_len * fixed_len)
                _tmp_segment = np.reshape(resampled_seg, newshape=[fixed_len, raw_len])
                resampled_segment = np.mean(_tmp_segment, axis=1)
                segments1.append(np.expand_dims(resampled_segment, axis=0))

            segments1 = normalize(np.concatenate(segments1, axis=0))
            segments0.append(segments1)
            raw_lengths0.append(np.array(raw_lengths1))

        return np.array(segments0), None, np.array(raw_lengths0), self.diagnose

    def detect_beats(self, lead):
        return np.array(detect_beats(self.raw_data[lead, 0, :], rate=self.fs))

    def detect_qrs(self, lead):
        return np.array(detect_qrs(self.raw_data[lead, 0, :], fs=self.fs))

    @property
    def heartaxis(self):
        try:
            I_lead_index = self.lead_dict['I']
        except KeyError:
            print("I lead not found in ecg record. ")
            return None
        try:
            aVF_lead_index = self.lead_dict['aVF']
        except KeyError:
            print("aVF lead not found in ecg record. ")
            return None
        I_signal = self.raw_data[I_lead_index, 0, :]
        aVF_signal = self.raw_data[aVF_lead_index, 0, :]
        axis_angle = []
        for i in range(len(self.beat_label)):
            if self.beat_label[i]['Ronset'] and self.beat_label[i]['Roffset']:
                sta = self.beat_label[i]['Ronset'][-1]
                end = self.beat_label[i]['Roffset'][0]
                sum_I = sum(I_signal[sta:end+1])
                sum_aVF = sum(aVF_signal[sta:end+1])
                angle = math.atan2(2 * sum_aVF, math.pi * sum_I) * 57.3
                axis_angle.append(angle)
        mean_angle = np.mean(np.array(axis_angle))
        return mean_angle

    @property
    def atrial_rate(self):
        count_p = 0
        for i in range(1, len(self.beat_label)):
            count_p += len(self.beat_label[i]['P'])
        sta = self.beat_label[0]['R'][0]
        end = self.beat_label[-1]['R'][0]
        dur = (end - sta) / float(self.fs)
        atrial_rate = count_p / float(dur) * 60
        return atrial_rate

    @property
    def ventricular_rate(self):
        count_r = 0
        for i in range(len(self.beat_label)-1):
            count_r += len(self.beat_label[i]['R'])
        sta = self.beat_label[0]['R'][0]
        end = self.beat_label[-1]['R'][0]
        dur = (end - sta) / float(self.fs)
        ventricular_rate = count_r / float(dur) * 60
        return ventricular_rate

    @property
    def pr_interval(self):
        pr_interval_list = []
        for i in range(len(self.beat_label)):
            if self.beat_label[i]['Ponset'] and self.beat_label[i]['Ronset']:
                pr_interval_list.append(abs(self.beat_label[i]['Ponset'][-1] - self.beat_label[i]['Ronset'][0]) / float(self.fs))
        pr_interval_list = np.array(pr_interval_list)
        pr_interval = np.mean(pr_interval_list)
        return pr_interval

    @property
    def qt_interval(self):
        qt_interval_list = []
        for i in range(len(self.beat_label)):
            if self.beat_label[i]['Ronset'] and self.beat_label[i]['Toffset']:
                qt_interval_list.append((self.beat_label[i]['Toffset'][0]-self.beat_label[i]['Ronset'][0])/float(self.fs))
        mean_qt_interval = np.mean(np.array(qt_interval_list))
        return mean_qt_interval

    @property
    def qtc_interval(self):
        qtc_interval_list = []
        for i in range(1, len(self.beat_label)-1):
            if self.beat_label[i]['Ronset'] and self.beat_label[i]['Toffset']:
                correct_rate = math.sqrt((self.beat_label[i+1]['R'][0]-self.beat_label[i]['R'][0])/float(self.fs))
                qtc_interval_list.append((self.beat_label[i]['Toffset'][0]-self.beat_label[i]['Ronset'][0])/float(self.fs)/correct_rate)
        qtc_interval_list = np.array(qtc_interval_list)
        mean_QTc_interval = np.mean(qtc_interval_list)
        return mean_QTc_interval

    @property
    def p_wave_duration(self):
        pdur_list = []
        for i in range(len(self.beat_label)):
            if self.beat_label[i]['Ponset'] and self.beat_label[i]['Poffset']:
                pdur_list.append(abs(self.beat_label[i]['Ponset'][0] - self.beat_label[i]['Poffset'][0]) / float(self.fs))
        pdur = np.mean(np.array(pdur_list))
        return pdur

    @property
    def qrs_duration(self):
        qrs_dur_list = []
        for i in range(len(self.beat_label)):
            if self.beat_label[i]['Ronset'] and self.beat_label[i]['Roffset']:
                qrs_dur_list.append(abs(self.beat_label[i]['Ronset'][0]-self.beat_label[i]['Roffset'][0])/float(self.fs))
        qrs_dur = np.mean(np.array(qrs_dur_list))
        return qrs_dur

    def rs_ratio(self, lead):
        lead_signal = self.raw_data[lead, 0, :]
        ratio_list = []
        for i in range(len(self.beat_label)):
            if self.beat_label[i]['Ronset'] and self.beat_label[i]['Roffset'] and self.beat_label[i]['Poffset'] and self.beat_label[i]['Tonset']:
                sta = self.beat_label[i]['Poffset'][-1]
                end = self.beat_label[i]['Tonset'][0]
                qonset = self.beat_label[i]['Ronset'][-1]
                soffset = self.beat_label[i]['Roffset'][0]
                if sta < qonset < soffset < end:
                    segsig = np.array(lead_signal[sta:end])
                    if segsig.max() < 0:
                        segsig -= 2 * segsig.max()
                    pos_peak_ind, neg_peak_ind, base_seg = qrs_peak_detect(segsig, qonset - sta, soffset - sta)
                    ratio = abs(np.mean(segsig[pos_peak_ind])) / abs(np.mean(segsig[neg_peak_ind]))
                    ratio_list.append(ratio)
        return np.mean(np.array(ratio_list))

    def r_amplitudes(self, lead):
        lead_signal = self.raw_data[lead, 0, :]
        r_list = []
        for i in range(len(self.beat_label)):
            if self.beat_label[i]['Ronset'] and self.beat_label[i]['Roffset'] and self.beat_label[i]['Poffset'] and self.beat_label[i]['Tonset']:
                sta = self.beat_label[i]['Poffset'][-1]
                end = self.beat_label[i]['Tonset'][0]
                qonset = self.beat_label[i]['Ronset'][-1]
                soffset = self.beat_label[i]['Roffset'][0]
                if sta < qonset < soffset < end:
                    segsig = np.array(lead_signal[sta:end])
                    if segsig.max() < 0:
                        segsig -= 2 * segsig.max()
                    pos_peak_ind, neg_peak_ind, base_seg = qrs_peak_detect(segsig, qonset - sta, soffset - sta)
                    r_amp = abs(np.mean(segsig[pos_peak_ind]))
                    r_list.append(r_amp)
        return np.array(r_list)

    def s_amplitudes(self, lead):
        lead_signal = self.raw_data[lead, 0, :]
        s_list = []
        for i in range(len(self.beat_label)):
            if self.beat_label[i]['Ronset'] and self.beat_label[i]['Roffset'] and self.beat_label[i]['Poffset'] and self.beat_label[i]['Tonset']:
                sta = self.beat_label[i]['Poffset'][-1]
                end = self.beat_label[i]['Tonset'][0]
                qonset = self.beat_label[i]['Ronset'][-1]
                soffset = self.beat_label[i]['Roffset'][0]
                if sta < qonset < soffset < end:
                    segsig = np.array(lead_signal[sta:end])
                    if segsig.max() < 0:
                        segsig -= 2 * segsig.max()
                    pos_peak_ind, neg_peak_ind, base_seg = qrs_peak_detect(segsig, qonset - sta, soffset - sta)
                    s_amp = abs(np.mean(segsig[neg_peak_ind]))
                    s_list.append(s_amp)
        return np.array(s_list)

    def qrs_areas(self, lead):
        lead_signal = self.raw_data[lead, 0, :]
        sum_qrs = list()
        for i in range(len(self.beat_label)):
            if self.beat_label[i]['Ronset'] and self.beat_label[i]['Roffset'] and self.beat_label[i]['Poffset'] and self.beat_label[i]['Tonset']:
                sta = self.beat_label[i]['Poffset'][-1]
                end = self.beat_label[i]['Tonset'][0]
                qonset = self.beat_label[i]['Ronset'][-1]
                soffset = self.beat_label[i]['Roffset'][0]
                if sta < qonset < soffset < end:
                    segsig = np.array(lead_signal[sta:end])
                    if segsig.max() < 0:
                        segsig -= 2 * segsig.max()
                    pos_peak_ind, neg_peak_ind, base_seg = qrs_peak_detect(segsig, qonset - sta, soffset - sta)
                    sum_seg = abs(np.sum(segsig[pos_peak_ind])) + abs(np.sum(segsig[neg_peak_ind]))
                    sum_qrs.append(sum_seg)
        return np.array(sum_qrs)

    def plot(self, duration=None, title=None, savepath=None):
        if len(self.lead_match) < 12:
            print("Error: Number of leads < 12.")
            return
        if title is None:
            title = self.path.split('/')[-1]
        data = np.squeeze(self.raw_data)
        if duration is not None:
            data = data[:, duration]
        data = data / 1000
        plot_12(data, title=title, sample_rate=self.fs, lead_index=self.lead_match)
        if savepath:
            save_as_png(savepath)


def distort(signals, labels, amps_p, amps_t):
    """
    Given p pos and t pos, augment p wave and t wave.
    :param signals:
    :param labels: shape = [lead, num, 8 or 9].
    :param amps_p:
    :param amps_t:
    :return:
    """
    assert len(signals.shape) == 3
    assert len(labels.shape) == 3
    segments0 = []
    labels0 = []
    label_len = labels.shape[2]
    assert label_len == 8 or label_len == 9
    for l in range(signals.shape[0]):
        segments1 = []
        labels1 = []
        for k in range(signals.shape[1]):
            signal = signals[l, k, :]
            label = labels[l, k, :]
            p = label[1]
            if label_len == 8:
                t = label[6]
            else:
                t = label[7]
            index_array = np.arange(0, len(signal))

            width_p_l = 2
            width_p_r = 0.5
            a_l_p = index_array[0:p]
            a_r_p = index_array[p:]
            center_l_p = max((label[1] + label[0]) / 2 - 5, 0)
            center_r_p = min((label[2] + label[1]) / 2 + 5, len(signal) - 1)
            a_l_p = 1 / (1 + np.exp(-(a_l_p - center_l_p) / width_p_l))
            a_r_p = 1 - 1 / (1 + np.exp(-(a_r_p - center_r_p) / width_p_r))

            width_t_l = 10
            width_t_r = 10
            a_l_t = index_array[0: t]
            a_r_t = index_array[t:]
            if label_len == 8:
                center_r_t = min((label[6] + label[7]) / 2 + 5, len(signal) - 1)
            else:
                center_r_t = min((label[7] + label[8]) / 2 + 5, len(signal) - 1)
            center_l_t = 2 * t - center_r_t
            a_l_t = 1 / (1 + np.exp(-(a_l_t - center_l_t) / width_t_l))
            a_r_t = 1 - 1 / (1 + np.exp(-(a_r_t - center_r_t) / width_t_r))

            for amp_p in amps_p:
                for amp_t in amps_t:
                    a_p = np.concatenate([a_l_p * amp_p, a_r_p * amp_p])
                    raw_segment_p = np.multiply(signal, a_p)
                    a_t = np.concatenate([a_l_t * amp_t, a_r_t * amp_t])
                    raw_segment_t = np.multiply(signal, a_t)
                    raw_segment = raw_segment_p + raw_segment_t + signal
                    segments1.append(np.expand_dims(raw_segment, axis=0))
                    labels1.append(np.expand_dims(label, axis=0))

        segments0.append(np.concatenate(segments1, axis=0))
        labels0.append(np.concatenate(labels1, axis=0))

    return np.array(segments0), np.array(labels0)
