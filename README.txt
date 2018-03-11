
Data are time series, represented by intervals between events in arbitrary time units. Each interval series is a row.
All time series has to be the same length, but the length is arbitrary. Shorter time series can be
extended by padding end with data points from start.

First item in each row (i.e. first column) is an integer category which the CNN is trained on.