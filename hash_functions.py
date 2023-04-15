import math


class HyperLogLog:
    """
    A class for the HyperLogLog algorithm for estimating the number of distinct elements in a set.
    """

    def __init__(self, b=10, sqr_num_buckets=0):
        """
        Initialize the HyperLogLog object.

        Args:
            b (int): The number of bits to use for the hash values.
            num_buckets (int) : The square root of number of buckets, each of them will count on their own.
        """
        self.b = b
        self.sqr_num_buckets = sqr_num_buckets
        self.num_buckets = sqr_num_buckets ^ 2  # number of buckets
        self.alpha = self.get_alpha(self.num_buckets)

        self.buckets = [0] * self.num_buckets

    def add_by_zeroes(self, zeroes, bucket_ind):
        """
        Add the count of zeroes of an element to the set.

        Args:
            zeroes (int): The number of leading zeroes of the element
            bucket_ind (int)
        """
        # Update the maximum number of leading zeros in the bucket
        self.buckets[bucket_ind] = max(self.buckets[bucket_ind], zeroes)

    def add_element(self, x):
        """
        Add an element to the set.

        Args:
            x (str): The element to be added to the counter
        """
        hash_code = bin(hash(x))
        bucket_ind = int(hash_code[len(hash_code) - self.sqr_num_buckets:], 2)

        for ind in range(1, len(hash_code) - self.sqr_num_buckets):
            if hash_code[ind] == b'1':
                self.add_by_zeroes(ind, bucket_ind)
                return

        # if all was zero
        self.add_by_zeroes(len(hash_code) - self.sqr_num_buckets - 1, bucket_ind)

    def estimate(self, correction_m):
        """
        Estimate the number of distinct elements in the set.

        Args:
            correction_m (int): A correction given by the caller determined by the use.

        Returns:
            The estimated number of distinct elements in the set.
        """
        # Calculate the harmonic mean of the bucket values
        harmonic_mean = self.alpha * (float(self.num_buckets) ** 2) / sum([2 ** -m for m in self.buckets])

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

    @staticmethod
    def count_leading_zeros(x):
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
    def get_alpha(num_buckets):
        """
        Calculate the alpha factor for the given number of buckets.

        Args:
            num_buckets (int): The number of buckets.

        Returns:
            The alpha factor.
        """
        return 0.7213 / (1 + 1.079 / num_buckets)


if __name__ == '__main__':
    hll = HyperLogLog(b=5)
    for i in word_list:
        hll.add(i)
    print(hll.estimate(), len(word_list))
