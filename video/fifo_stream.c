#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <signal.h>
#include <getopt.h>
#include <stdbool.h>

 
int main(int argc, char * const *argv)
{
	const char *infile = argv[1];
	const char *outfile = argv[2];
	int fd_in, fd_out;
	int opt;
	float rate = 0.0;
	bool exit_on_end = false;

	while ((opt = getopt(argc, argv, "r:e")) != -1) {
		switch (opt) {
		case 'r':
			rate = atof(optarg);
			break;
		case 'e':
			exit_on_end = true;
			break;
		}
	}

again:
	fd_in = open(infile, O_RDONLY);
	if (fd_in == -1) {
		perror(infile);
		exit(1);
	}

	unlink(outfile);
	if (mkfifo(outfile, 0644) != 0) {
		perror(outfile);
		exit(1);
	}

	signal(SIGPIPE,	SIG_IGN);

	printf("Waiting for connection ...\n");
	fd_out = open(outfile, O_WRONLY);
	if (fd_out == -1) {
		perror(outfile);
		exit(1);
	}

	printf("Streaming ...\n");
	while (1) {
		char buf[1024];
		ssize_t n = read(fd_in, buf, sizeof(buf));
		if (n <= 0) {
			if (exit_on_end) break;
			usleep(1000);
			continue;
		}
		if (write(fd_out, buf, n) == -1) {
			close(fd_out);
			close(fd_in);
			goto again;
		}
		if (rate != 0.0) {
			usleep(1.0e6/rate);
		}
	}

	printf("done\n");
	
	return 0;
}
