clear all
set linesize 255
set more off
set maxvar 120000

global ROOT ".." 
global PROCESSED_DATA "$ROOT/processed_data"
global TABLES "$ROOT/tables"

global OUTPUT "replace varwidth(25) mlabels(, titles) scalars(N r2_a r2_p N_clust) sfmt(%9.0fc %9.3fc %9.3fc %9.0fc) noobs b(3) t(2) starlevels(* 0.10 ** 0.05 *** 0.01) nogaps nocons"


// // cd "C:\Users\wenzh\Dropbox\Research\2025_Meme"
// import delimited using "pfm.csv", clear
//
// cap drop migration
// gen migration = 0
// replace migration = 1 if chain == "raydium" | chain == "pre_trump_raydium"
//
// gen period = "Pre-Trump"
// replace period = "Post-Trump" if chain == "pre_trump_pumpfun" | chain == "pre_trump_raydium"
//
// eststo clear
// eststo: logit migration launch_bundle sniper_bot volume_bot bot_comment_num [pw=weight], vce(robust)
// eststo: logit migration launch_bundle sniper_bot volume_bot bot_comment_num [pw=weight] if period == "Pre-Trump", vce(robust)
// eststo: logit migration launch_bundle sniper_bot volume_bot bot_comment_num [pw=weight] if period == "Post-Trump", vce(robust)
// esttab, ${OUTPUT}
// esttab using "reg_pfm.rtf", ${OUTPUT}


// import delimited using "$PROCESSED_DATA/reg_did_file/reg_volume_did.csv", clear
// keep if cohort <= 60
// save reg_volume_did, replace

// use reg_volume_did, clear
// gen number_of_traders = exp(log_number_of_traders)
//
// egen ct_fe = group(cohort time)
// egen cp_fe = group(cohort token)
//
// eststo clear
// // eststo: reghdfe log_number_of_traders c.treat##c.post [pw=weight], a(ct_fe cp_fe) vce(cluster cp_fe)
// eststo: ppmlhdfe number_of_traders c.treat##c.post [pw=weight], a(ct_fe cp_fe) vce(cluster cp_fe)
// esttab, ${OUTPUT}
// esttab using "reg_volume_did.rtf", ${OUTPUT}
//
//
// gen offset2 = 5 * floor(offset / 5)
//
// keep if cohort >= 5
// collapse (mean) number_of_traders [pw=weight], by(treat offset2)
//
// twoway (connected number_of_traders offset2 if treat == 0, msymbol(X) msize(medium)) ///
//        (connected number_of_traders offset2 if treat == 1, msymbol(O) msize(medium)) ///
//        , ///
//        xline(-2.5, lcolor(red) lpattern(dash)) ///
//        legend(order(2 "Treatment" 1 "Control")) ///
//        xlabel(, grid) ylabel(, grid) ///
//        xtitle("Offset") ytitle("Number of Traders") ///
//        title("Treatment Effects of Wash Trading Bot")













import delimited using "$PROCESSED_DATA/reg_did_file/reg_comment_did.csv", clear
keep if cohort <= 60
save reg_comment_did, replace

use reg_comment_did, clear
gen number_of_traders = exp(log_number_of_traders)

egen ct_fe = group(cohort time)
egen cp_fe = group(cohort token)

eststo clear
// eststo: reghdfe log_number_of_traders c.treat##c.post [pw=weight], a(ct_fe cp_fe) vce(cluster cp_fe)
eststo: ppmlhdfe number_of_traders c.treat##c.post [pw=weight], a(ct_fe cp_fe) vce(cluster cp_fe)
esttab, ${OUTPUT}
esttab using "reg_comment_did.rtf", ${OUTPUT}


// forval i = 5(-1)1 {
// 	gen treat__`i' = treat * (offset == -`i')
// }
// forval i=0/10 {
// 	gen treat_`i' = treat * (offset == `i')
// }
//
// drop treat__1
//
// eststo clear
// eststo: ppmlhdfe number_of_traders c.treat_* [pw=weight] if inrange(offset, -5, 10), a(ct_fe cp_fe) vce(cluster cp_fe)
// esttab, ${OUTPUT}

gen offset2 = 5 * floor(offset / 5)

keep if cohort >= 5
collapse (mean) number_of_traders [pw=weight], by(treat offset2)

twoway (connected number_of_traders offset2 if treat == 0, msymbol(X) msize(medium)) ///
       (connected number_of_traders offset2 if treat == 1, msymbol(O) msize(medium)) ///
       , ///
       xline(-2.5, lcolor(red) lpattern(dash)) ///
       legend(order(2 "Treatment" 1 "Control")) ///
       xlabel(, grid) ylabel(, grid) ///
       xtitle("Offset") ytitle("Number of Traders") ///
       title("Treatment Effects of Comment Bot")
