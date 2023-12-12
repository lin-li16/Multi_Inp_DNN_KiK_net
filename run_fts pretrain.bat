@REM call activate
@REM for /l %%a in (0, 1, 2) do (
@REM     for /l %%b in (1, 1, 3) do (
@REM         python main_fts.py --path inp_arg%%b_model_events --resultspath basic_%%binps_100events_fts%%a --model Basic_%%binp --epoch 1000 --lr 0.001 --batch 1024 --normalize standard --datapre data_preprocess_100events_logFTS%%a.pkl --kernel 17 --plots 1
@REM     )
@REM )
@REM pause

call activate
set sta=NIGH13
for %%a in (5, 10, 20, 30, 40, 50) do (
    python main_fts.py --path inp_arg3_model_events --resultspath %sta%_pretrain_train%%a --model Basic_3inp --pretrain basic_3inps_100events_fts0 --epoch 200 --kernel 15 --lr 0.001 --batch 4 --normalize standard --datapre %sta%_100events_logFTS0_train%%a.pkl --plots 0
    python main_fts.py --path inp_arg3_model_events --resultspath %sta%_no_pretrain_train%%a --model Basic_3inp --epoch 1000 --kernel 15 --lr 0.001 --batch 4 --normalize standard --datapre %sta%_100events_logFTS0_train%%a.pkl --plots 0
)
pause