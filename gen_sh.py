import sys
import os

patient = ['S01', 'S02', 'S06', 'S07', 'S08']
#patient = ['S01', 'S02', 'S03', 'S05', 'S06', 'S07', 'S08', 'S09']
#patient = ['S01']
distance = [100, 133, 166, 200, 233, 266, 300, 333, 366, 400]
tstep = [1, 8, 9, 11, 12, 14, 16]
#tstep = [2, 3, 4]
fid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
#fid = [0]

if len(sys.argv) != 3 :
    print('Usage: python3 gen_sh.py input_file output_file')
    exit(0);

rfile = open(sys.argv[1], 'r')
wfile = open(sys.argv[2], 'w')


for row in rfile:
    srow = row.split()
    if len(srow) == 17 :
       i = int(srow[16])
       print(srow[0], srow[1], srow[2], srow[3], srow[4], srow[5], srow[6], srow[7], srow[8], 
             srow[9], srow[10], srow[11], srow[12], srow[13], srow[14], srow[15], i+10, file =wfile)
       newrow = srow
    else :
        print('', file=wfile)
for d in distance :
    for t in tstep :
        for f in fid :
            print(newrow[0], newrow[1], newrow[2], newrow[3], newrow[4], newrow[5], d, t, newrow[8], 
                  newrow[9], newrow[10], newrow[11], newrow[12], newrow[13], newrow[14], newrow[15], f, file =wfile)
        print('', file=wfile)

rfile.close()
wfile.close()
