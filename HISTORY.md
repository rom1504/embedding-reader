## 1.4.1

* add retry in parquet numpy

## 1.4.0

* faster numpy parquet

## 1.3.0

* parquet numpy reader

## 1.2.2

* fix parallel_pieces default

## 1.2.1

* [Fix] get_file_list function fix

## 1.2.0

* provide consecutive ids in all cases

## 1.1.10

* try catch whole block in reader to avoid getting stuck in case of error
* Fix meta column assignment

## 1.1.9

* don't use threads in read_table in parquet reader

## 1.1.8

* specify dependencies minimal versions

## 1.1.7

* propagating exceptions from thread pools

## 1.1.6

* fix progress report in parquet reader

## 1.1.5

* Fix some edges cases and clean some codes

## 1.1.4

* fix numpy reader handling of empty files

## 1.1.3

* check the empty file edge case in both piece builder and parquet reader

## 1.1.2

* improve piece builder test and fix the function

## 1.1.1

* improve default amount of parallel pieces

## 1.1.0

* add show progress option
* use better default for max piece size

## 1.0.0

implementation: it works

* numpy and parquet parallel reader
* piece builder
* doc
* tests
