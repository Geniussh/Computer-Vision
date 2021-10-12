import os
andrew_id = None
if andrew_id is None:
    print('enter your andrew id in the file first!')
    exit()


def check_file(file):
    if os.path.isfile(file):
        return True
    else:
        print('{} not found!'.format(file))
        return False
    

if  (check_file('../'+andrew_id+'/LucasKanade.py') and \
     check_file('../'+andrew_id+'/LucasKanadeAffine.py') and \
     check_file('../'+andrew_id+'/SubtractDominantMotion.py') and \
     check_file('../'+andrew_id+'/InverseCompositionAffine.py') and \
     check_file('../'+andrew_id+'/testCarSequence.py') and \
     check_file('../'+andrew_id+'/testCarSequenceWithTemplateCorrection.py') and \
     check_file('../'+andrew_id+'/testGirlSequence.py') and \
     check_file('../'+andrew_id+'/testGirlSequenceWithTemplateCorrection.py') and \
     check_file('../'+andrew_id+'/testAntSequence.py') and \
     check_file('../'+andrew_id+'/testAerialSequence.py') and \
     check_file('../'+andrew_id+'/girlseqrects.npy') and \
     check_file('../'+andrew_id+'/girlseqrects-wcrt.npy') and \
     check_file('../'+andrew_id+'/carseqrects.npy') and \
     check_file('../'+andrew_id+'/carseqrects-wcrt.npy')):
    print('file check passed!')
else:
    print('file check failed!')

#modify file name according to final naming policy
#you should also include files for extra credits if you are doing them (this check file does not check for them)
#images should be be included in the report