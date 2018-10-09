#!/usr/bin/perl -w
#
# eval_morphemes_v2.pl [-partitions <nparts>] [-trace] \
#                      wordpairs_goldstd wordpairs_proposed \
#                      morphemeanalyses_goldstd morphemeanalyses_proposed
#
# Version 2.1
#
# `eval_morphemes_v2.pl' computes the precision, recall, and (evenly-weighted)
# F-measure of the proposed morpheme analyses of a set of words (in the file
# `morphemeanalyses_proposed') with respect to a gold-standard (in the file
# `morphemeanalyses_goldstd').
#
# Note that there is no satisfactory means of ensuring that the labels of the
# morphemes are the same in the gold-standard and the proposed analyses; e.g.,
# the gold standard may use the label "+3SG" to denote the third person present
# tense verb ending in English (as in "comes, opens, listens"), whereas some
# unsupervised algorithm that discovers this morpheme may label it, e.g., "s1".
# Therefore, in practice the evaluation is performed by sampling a set of word
# pairs from the gold-standard, such that each pair has at least one morpheme
# in common (e.g., the English words "feet" and "horses" have the plural
# morpheme in common). A comparison is then performed to the proposed analyses
# of the same words; recall is defined as the proportion of word pairs, the
# proposed analyses of which indeed have morphemes in common. Precision is
# calculated accordingly: a set of word pairs is sampled from the proposed
# analyses, such that in each pair there is at least one proposed morpheme in
# common. The result is compared to the gold-standard and a check is performed
# whether these words are supposed (according to the gold-standard) to have
# morphemes in common or not. F-measure is calculated as the harmonic mean
# of precision and recall.
#
# The word pairs sampled from the gold-standard are supplied in the file
# `wordpairs_goldstd' and the pairs sampled from the proposed analyses are
# supplied in the file `wordpairs_proposed'. These files are produced using
# the program `sample_word_pairs_v2.pl' from the gold-standard
# (`morphemeanalyses_goldstd') and proposed analyses
# (`morphemeanalyses_proposed'), respectively.
#
# `eval_morphemes.pl' takes the following arguments:
#
# <nparts>     The data is split into <nparts> partitions, and for each part
#              evaluation figures are calculated separately. The overall
#              evaluation figures are then calculated as the mean over
#              the partitions. This is useful if one wants to assess the
#              statistical significance of the results of one algorithm
#              in comparison to another. The default value of <nparts> is
#              one, i.e., no partitioning takes place.
#
# -trace       As the program progresses, each tested word pair is displayed
#              together with its evaluation result. Default behavior: off.
#
# wordpairs_goldstd          File containing word pairs sampled from the
#                            gold-standard using `sample_word_pairs_v2.pl'.
# wordpairs_proposed         File containing word pairs sampled from the
#                            proposed analyses using `sample_word_pairs_v2.pl'.
# morphemeanalyses_goldstd   File containing the gold-standard morpheme
#                            analyses of the words.
# morphemeanalyses_proposed  File containing the proposed morpheme analyses
#                            of the words.
#
# The format of the lines in the files `morphemeanalyses_goldstd' and
# `morphemeanalyses_proposed' is the following:
#
# <word><tab><morpheme1><space><morpheme2><space>... etc.
#
# The morpheme labels may contain any printable characters except whitespace
# and comma. E.g.,
#
# tenderfeet      tender_A foot_N +PL
#
# Several alternative morpheme analyses for the word can be given by
# separating the analyses using commas. E.g., verb & noun interpretation:
#
# dreams          dream_V +3SG, dream_V +PL
#
#
# Changes in version 2:
#
# * In the case of alternative analyses, the number of matching morphemes
#   is no longer the total number of hits over all analyses, but the 
#   maximum number of hits over the analyses.
#
# * The precision and recall points are normalized to one point per 
#   word in the word pairs file, instead of one point per word pair. 
#   This reduces the weight of complex words/analyses.
#
# Changes in version 2.1:
#
# * Evaluation aborts if analysis file contains excess whitespace 
#   (instead of reporting incorrect results).
#
# Original script (eval_morphemes.pl) by Mathias Creutz, Aug 18 2006,
# for EU PASCAL MorphoChallenge 2007.
#
# Edited by Sami Virpioja, Feb 09 2009, for EU PASCAL MorphoChallenge 2009.
# Edited by Sami Virpioja, Jun 29 2010, for EU PASCAL MorphoChallenge 2010.
#

# Program starts here

# Command line arguments

($me = $0) =~ s,^.*/,,;

$nparts = 1;	# default value
$trace = 0;	# default value;
$wpairs_goldstd = "";  # input data files
$wpairs_proposed = ""; # - " -
$goldstd_file = "";    # - " -
$proposed_file = "";   # - " -

while (@ARGV) {
    $arg = shift @ARGV;
    if ($arg eq "-partitions") {        # user-defined value for number of
	&usage() unless (@ARGV);        # partitions
	$nparts = shift @ARGV;
	&usage() unless ($nparts =~ m/^[1-9][0-9]*$/);
    }
    elsif ($arg eq "-trace") {		# trace progress of program
	$trace = 1;
    }
    elsif (!$wpairs_goldstd) {		# input data files...
	$wpairs_goldstd = $arg;
    }
    elsif (!$wpairs_proposed) {
	$wpairs_proposed = $arg;
    }
    elsif (!$goldstd_file) {
	$goldstd_file = $arg;
    }
    elsif (!$proposed_file) {
	$proposed_file = $arg;
    }
    else {
	&usage();
    }
}

# Check that all input files were provided
&usage() unless ($wpairs_goldstd && $wpairs_proposed &&
		 $goldstd_file && $proposed_file);

$| = 1 if ($trace); # hot piping

#
# Compare the word pairs sampled from the gold standard to the same
# word pairs in the proposed analyses (=> estimate of recall)
#

@recall_nexpected = ();		# Partition-wise storage of accumulated
@recall_nfound = ();		# stats for the computation of recall
@recall_nexpectedaffixes = ();  # - " -
@recall_nfoundaffixes = ();     # - " -

&compare($wpairs_goldstd, $proposed_file, \@recall_nexpected, \@recall_nfound,
	 \@recall_nexpectedaffixes, \@recall_nfoundaffixes);

#
# Compare the word pairs sampled from the proposed analyses to the same
# word pairs in the gold standard (=> estimate of precision)
#

@precision_nexpected = ();	   # Partition-wise storage of accumulated
@precision_nfound = ();		   # stats for the computation of precision
@precision_nexpectedaffixes = ();  # - " -
@precision_nfoundaffixes = ();     # - " -

&compare($wpairs_proposed, $goldstd_file, \@precision_nexpected,
	 \@precision_nfound, \@precision_nexpectedaffixes,
	 \@precision_nfoundaffixes);

# Compute precision, recall, and F-measure for each partition and the global
# scores as the average over the partitions. Note that since precision and
# recall are calculated from different sets of words, the partition-wise
# F-measures may be misleading: when the statistical significance of one
# method in comparison with another method is assessed, the partition-wise
# precision and recall values should be used in the first place, rather than
# F-measure.

my($i);
my($tot_recall) = 0;		# Accumulators of global stats
my($tot_recall_nonaffixes) = 0;
my($tot_recall_affixes) = 0;
my($tot_precision) = 0;
my($tot_precision_nonaffixes) = 0;
my($tot_precision_affixes) = 0;

die "Error ($me): Partition mismatch: expected $nparts partitions, applied " .
    scalar(@precision_nfound) . " on file \"$wpairs_goldstd\" and " .
    scalar(@recall_nfound) . " on file \"$wpairs_proposed\". You are probably".
    " trying to use too many partitions with respect to the amount of data " .
    "available.\n" if ((scalar(@precision_nfound) != $nparts) ||
		       (scalar(@recall_nfound) != $nparts));

foreach $i (0 .. $nparts-1) {
    print "#\n";
    printf("PART%d. Precision: %.2f%% (%.0f/%.0f); non-affixes: %.2f%% ". 
	   "(%.0f/%.0f); affixes: %.2f%% (%.0f/%.0f)\n", $i,
	   100*&div($precision_nfound[$i],$precision_nexpected[$i]),
	   $precision_nfound[$i], $precision_nexpected[$i],
	   100*&div($precision_nfound[$i]-$precision_nfoundaffixes[$i],
		    $precision_nexpected[$i]-$precision_nexpectedaffixes[$i]),
	   $precision_nfound[$i]-$precision_nfoundaffixes[$i],
	   $precision_nexpected[$i]-$precision_nexpectedaffixes[$i],
	   100*&div($precision_nfoundaffixes[$i],
		    $precision_nexpectedaffixes[$i]),
	   $precision_nfoundaffixes[$i], $precision_nexpectedaffixes[$i]);
    printf("PART%d. Recall:    %.2f%% (%.0f/%.0f); non-affixes: %.2f%% ". 
	   "(%.0f/%.0f); affixes: %.2f%% (%.0f/%.0f)\n", $i,
	   100*&div($recall_nfound[$i],$recall_nexpected[$i]),
	   $recall_nfound[$i], $recall_nexpected[$i],
	   100*&div($recall_nfound[$i]-$recall_nfoundaffixes[$i],
		    $recall_nexpected[$i]-$recall_nexpectedaffixes[$i]),
	   $recall_nfound[$i]-$recall_nfoundaffixes[$i],
	   $recall_nexpected[$i]-$recall_nexpectedaffixes[$i],
	   100*&div($recall_nfoundaffixes[$i],
		    $recall_nexpectedaffixes[$i]),
	   $recall_nfoundaffixes[$i], $recall_nexpectedaffixes[$i]);
    printf("PART%d. F-measure: %.2f%%; non-affixes: %.2f%%; ". 
	   "affixes: %.2f%%\n", $i,
	   200/(&div($precision_nexpected[$i],$precision_nfound[$i]) +
		     &div($recall_nexpected[$i],$recall_nfound[$i])),
	   200/(&div($precision_nexpected[$i]-$precision_nexpectedaffixes[$i],
		     $precision_nfound[$i]-$precision_nfoundaffixes[$i]) +
		&div($recall_nexpected[$i]-$recall_nexpectedaffixes[$i],
		     $recall_nfound[$i]-$recall_nfoundaffixes[$i])),
	   200/(&div($precision_nexpectedaffixes[$i],
		     $precision_nfoundaffixes[$i]) +
		&div($recall_nexpectedaffixes[$i],$recall_nfoundaffixes[$i])));
    
    # Accumulate globally
    $tot_recall += 100*&div($recall_nfound[$i],$recall_nexpected[$i]);
    $tot_recall_nonaffixes +=
	100*&div($recall_nfound[$i]-$recall_nfoundaffixes[$i],
		 $recall_nexpected[$i]-$recall_nexpectedaffixes[$i]);
    $tot_recall_affixes +=
	100*&div($recall_nfoundaffixes[$i],$recall_nexpectedaffixes[$i]);
    $tot_precision += 100*&div($precision_nfound[$i],$precision_nexpected[$i]);
    $tot_precision_nonaffixes +=
	100*&div($precision_nfound[$i]-$precision_nfoundaffixes[$i],
		 $precision_nexpected[$i]-$precision_nexpectedaffixes[$i]);
    $tot_precision_affixes +=
	100*&div($precision_nfoundaffixes[$i],$precision_nexpectedaffixes[$i]);
}

print "#\n";
printf("TOTAL. Precision: %.2f%%; non-affixes: %.2f%%; affixes: %.2f%%\n",
       $tot_precision/$nparts, $tot_precision_nonaffixes/$nparts,
       $tot_precision_affixes/$nparts);
printf("TOTAL. Recall:    %.2f%%; non-affixes: %.2f%%; affixes: %.2f%%\n",
       $tot_recall/$nparts, $tot_recall_nonaffixes/$nparts,
       $tot_recall_affixes/$nparts);
printf("TOTAL. F-measure: %.2f%%; non-affixes: %.2f%%; affixes: %.2f%%\n",
       2/(&div($nparts,$tot_precision) + &div($nparts,$tot_recall)),
       2/(&div($nparts,$tot_precision_nonaffixes) +
	  &div($nparts,$tot_recall_nonaffixes)),
       2/(&div($nparts,$tot_precision_affixes) +
	  &div($nparts,$tot_recall_affixes)));


# End of main program.

sub compare {
    # Arguments: (1) name of word pairs file, (2) name of file containing
    # word analyses, (3)-(6) references to lists, where the stats are
    # accumulated
    my($wpairs_file, $anals_file, $nexpected_part, $nfound_part,
       $naffixesexpected_part, $naffixesfound_part) = @_;

    print "#\n# Comparing word pairs in file \"$wpairs_file\" to analyses in ".
	"\"$anals_file\":\n#\n" if ($trace);

    # Read in the sampled word pairs

    my(@wordpairs) = ();     # Stores the lines of the word-pairs file
    my(%relevantword) = ();  # Look-up for all words occurring in any word pair

    my($line, @words);	# Help variables

    open(WPFILE, $wpairs_file) ||
	die "Error ($me): Unable to open file \"$wpairs_file\" for reading.\n";
    while ($line = <WPFILE>) {
	chomp $line;
	push @wordpairs, $line;
	# Put all the words occurring on this line into the word hash
	@words = split(" ", $line);
	$relevantword{$words[0]} = 1;	# The first ("main") word
	shift @words;
	# Next, every second item is a word, and every second is the
	# morpheme(s) (within brackets) through which the main word was linked
	# to each of the other words on the line:
	while (@words) {
	    $relevantword{$words[0]} = 1;  # A word paired with the main word
	    shift @words;
	    shift @words; # Skip the morpheme(s) item that follows the word
	}
    }
    close WPFILE;

    # From the file containing the analyses, read in only the analyses
    # of words that occur in any of the sampled pairs (= relevant words) 

    local(%anals) = (); # Stores morpheme analyses of the relevant words
			# (local variable => visible to subroutines!)

    my($word, $anal, $lnum);	# Help variables

    open(ANFILE, $anals_file) ||
	die "Error ($me): Unable to open file \"$anals_file\" for reading.\n";
    $lnum = 0;
    while ($line = <ANFILE>) {
	chomp $line;
        $lnum++;
	($word, $anal) = split(/\t/, $line);
        if ($word =~ / / || $anal =~ /  +/) {
            print STDERR "Excess whitespace detected in line $lnum:\n$line\n";
            die "Aborted: Incorrect format in file $anals_file\n";
        }
	if ($relevantword{$word}) {
	    $anal =~ s/, */, /g; # Ensure that each comma is followed by space
	    $anals{$word} = $anal;
	}
    }
    close ANFILE;

    # When evaluation statistics are gathered, keep track into which
    # partition each observation is going, using the following variables:
    my($partitionsize) = scalar(@wordpairs)/$nparts;
    my($crntpart) = 0;
    my($i) = 0;

    # Help variables:

    # Variables related to the processing of strings on the line of word pairs:
    my(@alts, @pairs, $pairno, $pairword, $linkmorphemestr, @linkmorphemes);
    my($morpheme);

    # The number of alternative analyses of the word: we normalize by this
    # value so that words having several alternative interpretations will
    # not get more weight than unambiguous words:
    my($nalts, $analyses, $analysis, $nmorphemes);

    # For each word pair, the number of morphemes they have in common:
    my($nmorphemesexpected, $nmorphemesfound);
    my($naffixmorphemesexpected, $naffixmorphemesfound);

    # Accumulated stats for one alternative analysis in the word pairs file
    my($naltexpected, $naltfound, $naltaffixesexpected, $naltaffixesfound);

    # Accumulated stats for one line in the word pairs file
    my($nexpected, $nfound, $naffixesexpected, $naffixesfound);

    foreach $line (@wordpairs) {
	$nexpected = 0; # Tot. number of common morphemes expected on this line
	$nfound = 0;	# Tot. number found
	$naffixesexpected = 0;	# Tot. num. affixes among morphemes expected
	$naffixesfound = 0;	# Tot. num. affixes among morphemes found
        ($word, $analyses) = split(" ", $line, 2);
        @alts = split(", ", $analyses);
        $nalts = scalar(@alts); # Number of alternative analyses
	print "# $word\t" if ($trace);
        foreach $analysis (@alts) {
            $naltexpected = 0;
            $naltfound = 0;
            $naltaffixesexpected = 0;
            $naltaffixesfound = 0;
            @pairs = split(" ", $analysis);
            $nmorphemes = scalar(@pairs) / 2;
            foreach $pairno (1 .. $nmorphemes) {
                $pairword = shift @pairs;        # The list contains words
                $linkmorphemestr = shift @pairs; # followed by link morphemes
                $linkmorphemestr =~ s/^\[//;
                $linkmorphemestr =~ s/\]$//;
                @linkmorphemes = split(/,/, $linkmorphemestr);
                # Collect statistics separately for affixes vs. other
                # morphemes: affixes are identified through an initial
                # plus sign in the label, e.g., "+PL", "+SG3". First,
                # count the number of affixes among the link
                # morphemes:
                $naffixmorphemesexpected = 0;
                foreach $morpheme (@linkmorphemes) {
                    $naffixmorphemesexpected++ if ($morpheme =~ m/^\+/);
                }

                if ($pairword eq "~") { # No word pair exists for this morpheme
                    $nmorphemesexpected = 0;
                    $nmorphemesfound = 0;
                }
                else {
                    $nmorphemesexpected = scalar(@linkmorphemes);
                    $nmorphemesfound =
                        &get_number_of_morphemes_in_common($word, $pairword);
                    $nmorphemesfound = $nmorphemesexpected
                        if ($nmorphemesfound > $nmorphemesexpected);
                    
                    unless ($nmorphemesfound == -1) { # Analyses missing
                        # Accumulate stats on this alternative: Here
                        # each word pair gets the same weight, so the
                        # number of matching morphemes and expected
                        # affixes are normalized:
                        $naltexpected += 1;
                        $naltfound += ($nmorphemesfound/$nmorphemesexpected);
                        # With the respect to the affixes, we don't
                        # know whether the matching morphemes are
                        # affixes or not (since the labels used by the
                        # users might be anything). Therefore, we just
                        # distribute the matching morphemes
                        # proportionally onto affixes and non-affixes:
                        $naltaffixesexpected += ($naffixmorphemesexpected /
                                                 $nmorphemesexpected);
                        $naltaffixesfound += ($nmorphemesfound /
                                              $nmorphemesexpected) *
                                              ($naffixmorphemesexpected /
                                               $nmorphemesexpected);
                    }
                }
                if ($trace) {
                    print " $pairword [$linkmorphemestr] " .
                        "(corr: ${nmorphemesfound}/$nmorphemesexpected)";
                }
            }
            if ($naltexpected > 0) {
                $nexpected += 1;
                $nfound += $naltfound / $naltexpected;
                $naffixesexpected += $naltaffixesexpected / $naltexpected;
                $naffixesfound += $naltaffixesfound / $naltexpected;
                if ($trace) {
                    $corr = $naltfound / $naltexpected;
                    print " (=> $naltfound / $naltexpected = $corr)";
                }
            } else {
                $nalts--;
            }
            if ($trace) {
                print ",";
            }
	}
	if ($trace) {
            if ($nexpected > 0) {
                $corr = $nfound / $nexpected;
                print " => $corr\n";
            } else {
                print "\n";
            }
        }

	# Accumulate the stats from the whole line into the accumulators
	# for the current partition. Normalize by number of alternative
	# morphological analyses of the word:
        if ($nalts > 0) {
            $nexpected_part->[$crntpart] += $nexpected/$nalts;
            $nfound_part->[$crntpart] += $nfound/$nalts;
            $naffixesexpected_part->[$crntpart] += $naffixesexpected/$nalts;
            $naffixesfound_part->[$crntpart] += $naffixesfound/$nalts;
        }
	if ($trace) {
            print "# ".$nexpected_part->[$crntpart]." ".
                $nfound_part->[$crntpart]."\n";
            print "# ".$naffixesexpected_part->[$crntpart]." ".
                $naffixesfound_part->[$crntpart]."\n";
        }

	# Move on to next partition?
	$i++;
	if ($i >= $partitionsize) {
	    $i = 0;
	    $crntpart++;
	}
    }
}

# Subroutine returning the number of morphemes (type count) that two words
# have in common: if the words have many alternative analyses, the best 
# match between the analyses is returned.
sub get_number_of_morphemes_in_common {

    my($word1, $word2) = @_;

    # Return -1 if analyses of either word is unknown:
    return -1 unless ((defined $anals{$word1}) && (defined $anals{$word2}));

    my(@alts1, @alts2);
    my($i1, $i2);
    my($n, $maxn);

    @alts1 = split(", ", $anals{$word1});
    @alts2 = split(", ", $anals{$word2});
    $maxn = 0;
    # Check for each pair of alternative analysis
    for ($i1 = 0; $i1 < scalar(@alts1); $i1++) {
        for ($i2 = 0; $i2 < scalar(@alts2); $i2++) {
            $n = &get_number_of_common($alts1[$i1], $alts2[$i2]);
            if ($n > $maxn) {
                $maxn = $n;
            }
        }
    }
    return $maxn; # Return maximum number of matching types
}

# Subroutine returning the number of morphemes (type count) that two 
# analyses have in common.
sub get_number_of_common {
    my($anal1, $anal2) = @_;
    my(%morphemes1) = ();   # Hash containing the morphemes of word1
    my(%matches) = ();	    # Hash containing morphemes common to both words
    my($m);
    # Word1:
    foreach $m (split(/ /, $anal1)) {
	$morphemes1{$m} = 1;
    }
    # Word2: intersection of morphemes
    foreach $m (split(/ /, $anal2)) {
	$matches{$m} = 1 if ($morphemes1{$m});
    }
    return scalar(keys %matches);   # Return number of matching morpheme types
}

# Division that can handle division by zero
sub div {
    my($num, $denom) = @_;
    if ($denom == 0) {
	return 1 if ($num == 0);
	return 1e+100;
    }
    return $num/$denom;
}

sub usage {
    die "Usage: $me [-partitions <nparts>] [-trace] wordpairs_goldstd " .
	"wordpairs_proposed morphemeanalyses_goldstd " .
	"morphemeanalyses_proposed\n";
}
