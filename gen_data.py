import sys
import os

patient = ['S01', 'S02', 'S06', 'S07', 'S08']
#patient = ['S01', 'S02', 'S03', 'S05', 'S06', 'S07', 'S08', 'S09']
#patient = ['S01']
distance = [100, 133, 166, 200, 233, 266, 300, 333, 366, 400]
tstep = [2, 3, 4, 5, 6, 7, 10]
#tstep = [2, 3, 4]
fid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#fid = [0]

for p in patient : 
    wfile1 = open(p + '_ta.dat', 'w')
    for t in tstep:
        wfile = open(p + '_t'+str(t) + '.dat', 'w')
        tc10k = 0
        tc200 = 0
        tc133 = 0
        tc66 = 0
        tc0 = 0
        for d in distance :
            accuracy = 0.0
            sensitivity = 0
            specificity = 0
            gmean = 0
            precision = 0
            recall = 0
            fmeasure = 0

            c10k = 0
            c200 = 0
            c133 = 0
            c66 = 0
            c0 = 0

            totd = 0
            totad = 0
            totfad = 0
            totratio = 0.0

            for f in fid :
                fname = p +'_' +  str(d) + '_' + str(t)+'_SimpleRNN_' + str(f) + '_output.txt' 
                file = open(fname, 'r')

                for row in file:
                    srow = row.split()
#                    print(row)
#                    print(srow[0], ' XX ', srow[1])
                    if srow[0] == 'sensitivity:' : 
                        sensitivity += float(srow[1])
                    elif srow[0] == 'specificity:' :
                        specificity += float(srow[1])
                    elif srow[0] == 'accuracy:' :
                        accuracy += float(srow[1])
                    elif srow[0] == 'G-mean:' :
                        gmean += float (srow[1])
                    elif srow[0] == 'precision' :
                        precision += float(srow[2])
                    elif srow[0] == 'Recall:' :
                        recall += float(srow[1])
                    elif srow[0] == 'f-measure:' :
                        fmeasure += float(srow[1])
                    elif srow[0] == 'episode' :
                        i = int(srow[9])
#                        print('i = ', i)
                        if i == 10000 :
                            c10k += 1
                        elif i >= 200 :
                            c200 += 1
                        elif i >=133 :
                            c133 += 1
                        elif i >= 66 :
                            c66 += 1
                        elif i >= 0 :
                            c0 += 0
                    elif srow[0] == 'total' and srow[1] == 'duration:' :
                        totd += int(srow[2])
                        totad += int(srow[7])
                    elif srow[0] == 'total' and srow[1] == 'false' :
                        totfad += int(srow[4])
                        totratio += float(srow[7])
                file.close()
            num = len(fid)
            totratio /= num
            totfad /= num
            totad /= num
            totd /= num
            aa1  = (c0+c66+c133+c200) / num
            aa2 = (c66+c133+c200) / num
            aa3 = (c133+c200) / num
            aa4 = c200 / num
            c0 = aa1
            c66 = aa2
            c133 = aa3
            c200 = aa4
            c10k = c10k / num
            tc10k += c10k
            tc200 += c200
            tc133 += c133
            tc66 += c66
            tc0 += c0

            fmeasure /= num
            recall /= num
            precision /= num
            gmean /= num
            specificity /= num
            sensitivity /= num
            accuracy /= num
            afstring = '{0:3d} {1:5.3f} {2:5.3f} {3:5.3f} {4:5.3f} {5:5.3f} {6:5.3f} {7:5.3f} {8:4.1f} {9:4.1f} {10:4.1f} {11:4.1f} {12:4.1f} {13:6.1f} {14:6.1f} {15:6.1f} {16:4.2f}' 

#            print('{0:3d} {1:5.3f} '.format(d, accuracy), file=wfile)
            print(afstring.format(d, accuracy, sensitivity, specificity, gmean, precision, recall, fmeasure, c10k, c0, c66, c133, c200, totd, totad, totfad, totratio), file=wfile)
        wfile.close()
        tc10k /= len(distance)
        tc200 /= len(distance)
        tc133 /= len(distance)
        tc66 /= len(distance)
        tc0 /= len(distance)

        print('{0:3d} {1:5.3f} {2:5.3f} {3:5.3f} {4:5.3f} {5:5.3f}'.format(t,  tc10k, tc0, tc66, tc133, tc200), file=wfile1)
    wfile1.close()

for p in patient : 
    wfile1 = open(p + '_da.dat', 'w')
    for d in distance :
        wfile = open(p + '_d'+str(d) + '.dat', 'w')
        dc10k = 0
        dc200 = 0
        dc133 = 0
        dc66 = 0
        dc0 = 0

        for t in tstep:
            accuracy = 0.0
            sensitivity = 0
            specificity = 0
            gmean = 0
            precision = 0
            recall = 0
            fmeasure = 0

            c10k = 0
            c200 = 0
            c133 = 0
            c66 = 0
            c0 = 0

            totd = 0
            totad = 0
            totfad = 0
            totratio = 0.0

            for f in fid :
                fname = p +'_' +  str(d) + '_' + str(t)+'_SimpleRNN_' + str(f) + '_output.txt' 
                file = open(fname, 'r')

                for row in file:
                    srow = row.split()
#                    print(row)
#                    print(srow[0], ' XX ', srow[1])
                    if srow[0] == 'sensitivity:' : 
                        sensitivity += float(srow[1])
                    elif srow[0] == 'specificity:' :
                        specificity += float(srow[1])
                    elif srow[0] == 'accuracy:' :
                        accuracy += float(srow[1])
                    elif srow[0] == 'G-mean:' :
                        gmean += float (srow[1])
                    elif srow[0] == 'precision' :
                        precision += float(srow[2])
                    elif srow[0] == 'Recall:' :
                        recall += float(srow[1])
                    elif srow[0] == 'f-measure:' :
                        fmeasure += float(srow[1])
                    elif srow[0] == 'episode' :
                        i = int(srow[9])
#                        print('i = ', i)
                        if i == 10000 :
                            c10k += 1
                        elif i >= 200 :
                            c200 += 1
                        elif i >=133 :
                            c133 += 1
                        elif i >= 66 :
                            c66 += 1
                        elif i >= 0 :
                            c0 += 0
                    elif srow[0] == 'total' and srow[1] == 'duration:' :
                        totd += int(srow[2])
                        totad += int(srow[7])
                    elif srow[0] == 'total' and srow[1] == 'false' :
                        totfad += int(srow[4])
                        totratio += float(srow[7])
                file.close()
            num = len(fid)
            totratio /= num
            totfad /= num
            totad /= num
            totd /= num
            aa1  = (c0+c66+c133+c200) / num
            aa2 = (c66+c133+c200) / num
            aa3 = (c133+c200) / num
            aa4 = c200 / num
            c0 = aa1
            c66 = aa2
            c133 = aa3
            c200 = aa4
            c10k = c10k / num
            dc10k += c10k
            dc200 += c200
            dc133 += c133
            dc66 += c66
            dc0 += c0

            fmeasure /= num
            recall /= num
            precision /= num
            gmean /= num
            specificity /= num
            sensitivity /= num
            accuracy /= num
            afstring = '{0:3d} {1:5.3f} {2:5.3f} {3:5.3f} {4:5.3f} {5:5.3f} {6:5.3f} {7:5.3f} {8:4.1f} {9:4.1f} {10:4.1f} {11:4.1f} {12:4.1f} {13:6.1f} {14:6.1f} {15:6.1f} {16:4.2f}' 

#            print('{0:3d} {1:5.3f} '.format(d, accuracy), file=wfile)
            print(afstring.format(t, accuracy, sensitivity, specificity, gmean, precision, recall, fmeasure, c10k, c0, c66, c133, c200, totd, totad, totfad, totratio), file=wfile)
        wfile.close()
        dc10k /= len(tstep)
        dc200 /= len(tstep)
        dc133 /= len(tstep)
        dc66 /= len(tstep)
        dc0 /= len(tstep)

        print('{0:3d} {1:5.3f} {2:5.3f} {3:5.3f} {4:5.3f} {5:5.3f}'.format(d,  dc10k, dc0, dc66, dc133, dc200), file=wfile1)
    wfile1.close()
