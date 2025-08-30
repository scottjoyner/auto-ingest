for f in *_FRA.mp4; do echo "file '$f'" >> mylist.txt; done
ffmpeg -f concat -safe 0 -i mylist.txt -c copy output.mp4
rm mylist.txt
ffmpeg -f concat -safe 0 -i mylist.txt -c copy hbd_26.mp4
ffmpeg -f concat -safe 0 -i mylist.txt -vf "scale=640:360" -b:v 800k -b:a 128k passengered.mp4

ffmpeg -i almost_macked_for_real.mp4 -vf scale=1280:720 -b:v 1M -r 24 -c:v libx265 -crf 28 -c:a aac -b:a 128k almost_macked_for_real_small.mp4
/media/deathstar/8TBHDD/fileserver/movies/almost_macked.mp4