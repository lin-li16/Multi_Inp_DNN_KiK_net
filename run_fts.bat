call activate
for /l %%a in (0, 1, 2) do (
    @REM for /l %%b in (1, 1, 3) do (
    python main_fts.py --path inp_arg2_model1_events --resultspath basic_2inps_100events_fts%%a --model Basic_2inp_1 --epoch 1000 --lr 0.001 --batch 2048 --normalize standard --datapre data_preprocess_100events_logFTS%%a.pkl --kernel 15 --plots 1
    @REM )
)
pause

@REM call activate
@REM for %%a in (11, 13, 15, 17, 19) do (
@REM     for %%b in (256, 512, 1024, 2048, 4096) do (
@REM         for %%c in (0.1, 0.01, 0.001, 0.0001) do (
@REM             python main_fts.py --path Hyper-parameters --resultspath kernel%%a_batch%%b_lr%%c --model Basic_3inp --epoch 1000 --kernel %%a --lr %%c --batch %%b --dropout 0.0 --normalize standard --datapre data_preprocess_100events_logFTS0.pkl --plots 0
@REM         )
@REM     )
@REM )
@REM pause