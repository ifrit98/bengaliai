#!/usr/bin/Rscript


parser <- Xmisc::ArgumentParser$new()

parser$add_argument(
  '--projdir', type = 'character',
  default = '/home/jason/internal/bengali',
  help = 'project directory to launch the training run from'
)

parser$add_argument(
  '--rundir', type = 'character',
  default = '/home/jason/internal/runs', 
  help = 'run directory to save results'
)

args = parser$get_args()

setwd(args$projdir)
options(tfruns.runs_dir = args$rundir)
source("flags.R")

rundir = tfruns::unique_run_dir(seconds_scale = 3)

system(paste0(
  'rsync -az --exclude=runs, --exclude=logs',
  ' --exclude=.Rproj.user, --exclude=.git',
  ' --exclude=plots --exclude=logs --exclude=data',
  ' --exclude=lit ./ ', rundir))

# manually copy model file to model.R in toplevel of rsync'd dir for `restore_model()`
system(paste0('cp ', FLAGS$model, ' ', rundir, '/model.R'))
system(paste0('echo run_dir: ', rundir))

setwd(rundir)

system(paste0('\n tfrun ', rundir))
  