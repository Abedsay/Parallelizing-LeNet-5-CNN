CC=mpic++
CFLAGS=-w -O1
TARGET=main
OBJS=main.o lenet.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) -lm

main.o: main.c lenet.h
	$(CC) $(CFLAGS) -c main.c

lenet.o: lenet.c lenet.h
	$(CC) $(CFLAGS) -c lenet.c

clean:
	rm -f $(TARGET) $(OBJS)
