void cuav_process(const uint8_t *buffer, uint32_t size, const char *filename, const char *linkname, const struct timeval *tv, bool halfres);

void *mm_alloc(uint32_t size);
void mm_free(void *ptr, uint32_t size);

