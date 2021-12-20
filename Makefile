CC = nvcc
FILES = reduce_many.o reduce_local.o main.o

all: PROGRAM

PROGRAM: ${FILES}
	${CC} ${FILES} -o prog

main.o: main.cu
	${CC} -o main.o -c main.cu

reduce_many.o: reduce_many.cu reduce_many.h
	${CC} -o reduce_many.o -c reduce_many.cu

reduce_local.o: reduce_local.cu reduce_local.h
	${CC} -o reduce_local.o -c reduce_local.cu

clean:
	rm -f ${FILES} prog
