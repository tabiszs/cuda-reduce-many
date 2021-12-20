CC = nvcc
FILES = reduce_many.o reduce_many_old.o reduce_local.o tests.o main.o
CFLAGS =  -std=c++11

all: PROGRAM

PROGRAM: ${FILES}
	${CC} ${FILES} -o prog

main.o: main.cu main.h
	${CC} -o main.o -c main.cu ${CFLAGS}

tests.o: tests.cpp tests.h
	${CC} -o tests.o -c tests.cpp ${CFLAGS}

reduce_many.o: reduce_many.cu reduce_many.h
	${CC} -o reduce_many.o -c reduce_many.cu

reduce_many_old.o: reduce_many_old.cu reduce_many_old.h
	${CC} -o reduce_many_old.o -c reduce_many_old.cu

reduce_local.o: reduce_local.cu reduce_local.h
	${CC} -o reduce_local.o -c reduce_local.cu

clean:
	rm -f ${FILES} prog
