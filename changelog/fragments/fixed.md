`Animation` no longer leaks an open file handle per frame when building a GIF; the frame images are now read into memory and their files closed immediately.
