library(dplyr)
library(readr)

pa = "nhanes"
fn = c("DEMO_J.csv.gz", "BMX_J.csv.gz", "BPX_J.csv.gz")

df = NULL
for (f in fn) {
    dx = read_csv(file.path(pa, f))
    if (is.null(df)) {
        df = dx
    } else {
        df = left_join(df, dx, "SEQN")
    }
}

df = df %>% mutate(RIAGENDR=recode(df$RIAGENDR, `1`="M", `2`="F"))

df = df %>% filter(RIDAGEYR >= 18)
