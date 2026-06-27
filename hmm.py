import math


class HMM(object):
    def __init__(self, N, M):
        self.N = N
        self.M = M

        self.A = [[0.0 for _ in range(N)] for _ in range(N)]
        self.B = [[0.0 for _ in range(M)] for _ in range(N)]
        self.Pi = [0.0 for _ in range(N)]

    def train(self, word_lists, tag_lists, word2id, tag2id):
        assert len(tag_lists) == len(word_lists)

        for tag_list in tag_lists:
            seq_len = len(tag_list)
            for i in range(seq_len - 1):
                current_tagid = tag2id[tag_list[i]]
                next_tagid = tag2id[tag_list[i + 1]]
                self.A[current_tagid][next_tagid] += 1.0
        self._normalize_rows(self.A)

        for tag_list, word_list in zip(tag_lists, word_lists):
            assert len(tag_list) == len(word_list)
            for tag, word in zip(tag_list, word_list):
                tag_id = tag2id[tag]
                word_id = word2id[word]
                self.B[tag_id][word_id] += 1.0
        self._normalize_rows(self.B)

        for tag_list in tag_lists:
            init_tagid = tag2id[tag_list[0]]
            self.Pi[init_tagid] += 1.0
        self._normalize_vector(self.Pi)

    def test(self, word_lists, word2id, tag2id):
        pred_tag_lists = []
        for word_list in word_lists:
            pred_tag_lists.append(self.decoding(word_list, word2id, tag2id))
        return pred_tag_lists

    def decoding(self, word_list, word2id, tag2id):
        log_A = self._log_matrix(self.A)
        log_B = self._log_matrix(self.B)
        log_Pi = self._log_vector(self.Pi)

        seq_len = len(word_list)
        viterbi = [[float("-inf") for _ in range(seq_len)] for _ in range(self.N)]
        backpointer = [[-1 for _ in range(seq_len)] for _ in range(self.N)]

        start_wordid = word2id.get(word_list[0])
        if start_wordid is None:
            start_emissions = self._uniform_log_probs()
        else:
            start_emissions = [log_B[state_id][start_wordid] for state_id in range(self.N)]

        for state_id in range(self.N):
            viterbi[state_id][0] = log_Pi[state_id] + start_emissions[state_id]

        for step in range(1, seq_len):
            wordid = word2id.get(word_list[step])
            if wordid is None:
                emissions = self._uniform_log_probs()
            else:
                emissions = [log_B[state_id][wordid] for state_id in range(self.N)]

            for current_state in range(self.N):
                best_prev_state = 0
                best_score = float("-inf")
                for prev_state in range(self.N):
                    score = viterbi[prev_state][step - 1] + log_A[prev_state][current_state]
                    if score > best_score:
                        best_score = score
                        best_prev_state = prev_state
                viterbi[current_state][step] = best_score + emissions[current_state]
                backpointer[current_state][step] = best_prev_state

        best_last_state = max(range(self.N), key=lambda state_id: viterbi[state_id][seq_len - 1])
        best_path = [best_last_state]
        for back_step in range(seq_len - 1, 0, -1):
            best_last_state = backpointer[best_last_state][back_step]
            best_path.append(best_last_state)

        id2tag = {id_: tag for tag, id_ in tag2id.items()}
        return [id2tag[state_id] for state_id in reversed(best_path)]

    @staticmethod
    def _normalize_rows(matrix):
        for row in matrix:
            for i, value in enumerate(row):
                if value == 0.0:
                    row[i] = 1e-10
            total = sum(row)
            for i, value in enumerate(row):
                row[i] = value / total

    @staticmethod
    def _normalize_vector(vector):
        for i, value in enumerate(vector):
            if value == 0.0:
                vector[i] = 1e-10
        total = sum(vector)
        for i, value in enumerate(vector):
            vector[i] = value / total

    @staticmethod
    def _log_matrix(matrix):
        return [[math.log(value) for value in row] for row in matrix]

    @staticmethod
    def _log_vector(vector):
        return [math.log(value) for value in vector]

    def _uniform_log_probs(self):
        uniform_prob = math.log(1.0 / self.N)
        return [uniform_prob for _ in range(self.N)]
