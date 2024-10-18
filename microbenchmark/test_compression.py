import lz4.frame

# Test with redundant data
redundant_data = b'A' * 1000  # A string with high redundancy
compressed_redundant = lz4.frame.compress(redundant_data, compression_level=0)
compressed_redundant_low = lz4.frame.compress(redundant_data, compression_level=0)

print(f"Redundant Data Sizes: {len(redundant_data)}, {len(compressed_redundant)}, {len(compressed_redundant_low)}")

# Test with less redundant data
non_redundant_data = b'This is some text that varies quite a bit, making it harder to compress effectively.' * 10
compressed_non_redundant = lz4.frame.compress(non_redundant_data, compression_level=9)
compressed_non_redundant_low = lz4.frame.compress(non_redundant_data, compression_level=1)

print(f"Non-Redundant Data Sizes: {len(non_redundant_data)}, {len(compressed_non_redundant)}, {len(compressed_non_redundant_low)}")
