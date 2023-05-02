import math
import mmh3


class LogLog:
    """
    A class for the HyperLogLog algorithm for estimating the number of distinct elements in a set.
    """

    def __init__(self, lg_num_buckets=0):
        """
        Initialize the HyperLogLog object.

        Args:
            lg_num_buckets (int) : The square root of number of buckets, each of them will count on their own.
        """
        self.lg_num_buckets = lg_num_buckets
        self.num_buckets = 2 ** self.lg_num_buckets  # number of buckets
        self.alpha = self.get_alpha(self.num_buckets)

        self.buckets = [0] * self.num_buckets

    def __str__(self):
        return f"bucket's content: {sorted(self.buckets)}"

    def add_by_zeros(self, zeros, bucket_ind) -> None:
        """
        Add the count of zeros of an element to the set.

        Args:
            zeros (int): The number of leading zeros of the element
            bucket_ind (int)
        """
        # Update the maximum number of leading zeros in the bucket
        self.buckets[bucket_ind] = max(self.buckets[bucket_ind], zeros + 1)

    def add_element(self, x) -> None:
        """
        Add an element to the set.

        Args:
            x (str): The element to be added to the counter
        """
        # get hash code and delete if there is an unnecessary minus
        hash_code = bin(abs(mmh3.hash(x)))

        bucket_ind = int(hash_code[3:self.lg_num_buckets + 3], 2)

        for ind in range(len(hash_code) - 1, self.lg_num_buckets + 3, -1):
            if hash_code[ind] == '1':
                self.add_by_zeros(len(hash_code) - ind - 1, bucket_ind)

                return

        # if all was zero
        self.add_by_zeros(len(hash_code) - self.lg_num_buckets - 4, bucket_ind)

    def estimate(self, correction_m=1) -> int:
        """
        Estimate the number of distinct elements in the set.
        Very accurate for stable hashes.

        Args:
            correction_m (int): A correction given by the caller determined by the use.

        Returns:
           The estimated number of distinct elements in the set.
        """
        # Calculate the harmonic mean of the bucket values
        harmonic_mean = self.alpha * (self.num_buckets ** 2) / sum([2 ** -bucket for bucket in self.buckets])

        # Use the empirical correction for small cardinalities
        if harmonic_mean <= 2.5 * self.num_buckets:
            num_zero_buckets = float(self.buckets.count(0))
            if num_zero_buckets != 0:
                corrected_mean = self.num_buckets * math.log(float(self.num_buckets) / num_zero_buckets)
            else:
                corrected_mean = harmonic_mean
        elif harmonic_mean > 1 / 30 * 2 ** 64:
            corrected_mean = -(2 ** 64) * math.log(1 - harmonic_mean / 2 ** 64)
        else:
            corrected_mean = harmonic_mean

        return int(corrected_mean * correction_m)

    def hyper_estimate(self, correction_m=1):
        """
        This estimate returns a much more conservative value that fits more for uncertainty
        in the hashes

        Args:
            correction_m (int): A correction given by the caller determined by the use.

        Returns:
           The estimated number of distinct elements in the set.
        """
        sorted_buckets = sorted(self.buckets)
        new_len = round(len(self.buckets) * 0.7)
        # take the 70 lowest values of buckets
        buckets_70 = sorted_buckets[:new_len]

        harmonic_mean = self.alpha * (new_len ** 2) / sum([2 ** -bucket for bucket in buckets_70])

        # Use the empirical correction for small cardinalities
        if harmonic_mean <= 2.5 * new_len:
            num_zero_buckets = float(buckets_70.count(0))
            if num_zero_buckets != 0:
                corrected_mean = new_len * math.log(float(new_len) / num_zero_buckets)
            else:
                corrected_mean = harmonic_mean
        elif harmonic_mean > 1 / 30 * 2 ** 64:
            corrected_mean = -(2 ** 64) * math.log(1 - harmonic_mean / 2 ** 64)
        else:
            corrected_mean = harmonic_mean

        return int(corrected_mean * correction_m)

    @staticmethod
    def count_leading_zeros(x) -> int:
        """
        Count the number of leading zeros in a binary number.

        Args:
            x (int): The number to count the leading zeros of.

        Returns:
            The number of leading zeros in the binary representation of the number.
        """
        if x == 0:
            return 0

        p = 0
        while (x >> p) & 1 == 0:
            p += 1

        return p + 1

    @staticmethod
    def get_alpha(num_buckets) -> float:
        """
        Calculate the alpha factor for the given number of buckets.

        Args:
            num_buckets (int): The number of buckets.

        Returns:
            The alpha factor.
        """
        return 0.7213 / (1 + 1.079 / num_buckets)


if __name__ == '__main__':
    # test of the algorithm
    with open("../words.txt", 'r') as file:
        word_list = file.read().split()

    hll = LogLog(lg_num_buckets=7)

    for i in word_list:
        hll.add_element(i)
    print(f"{hll.hyper_estimate()}, {hll.estimate()}, {len(set(word_list))}")
