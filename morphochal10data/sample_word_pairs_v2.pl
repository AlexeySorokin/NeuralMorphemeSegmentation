#!/usr/bin/perl -w
#
# sample_word_pairs_v2.pl [-n <nwords>] [-rand <randseed>]
#                         [-refwords <wordlist_file>] < inputdata
#
# `sample_word_pairs_v2.pl' takes a list of words and their proposed morpheme
# analyses as input data. The program selects <nwords> words by random,
# and for each morpheme in these words, another word that contains the same
# morpheme is selected by random (if such a word exists). As a result, the
# program will create a set of word pairs, where each pair has at least one
# morpheme in common. The list of word pairs is used as input to the
# program `eval_morphemes_v2.pl', which computes the precision, recall, and
# F-measure of the proposed morpheme analyses with respect to an existing
# gold standard.
#
# Arguments:
#
# <nwords>         The number of words sampled (default value 100, if omitted)
# <randseed>       Random seed (default value 0, if omitted)
# <wordlist_file>  A file containing words. If this option is given, any words
#                  in the input data that are not in this word list are
#                  skipped.
#                  This is a good thing if the vocabulary in the input data
#                  and that of the reference (gold standard) only overlap in
#                  part; it is of no use to select word pairs for evaluation,
#		   such that either word is missing from the reference.
#
# The format of the lines in the input data is the following:
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
# The format of the optional reference word list file can be the same:
# the relevant thing is that each line contains one word; everything
# from the first white space until the end of the line is ignored.
#
# Changes in version 2:
# * Several alternative analyses of the selected word can no longer affect 
#   crosswise. I.e., the common morphemes selected independently for each 
#   analysis.
#
# Original script (sample_word_pairs.pl) by Mathias Creutz, Aug 17 2006,
# for EU PASCAL MorphoChallenge 2007.
#
# Edited by Sami Virpioja, Feb 09 2009, for EU PASCAL MorphoChallenge 2009.
#

# Program starts here

# Read command line arguments

($me = $0) =~ s,^.*/,,;

$nwords = 100;	# default value
$randseed = 0;	# default value;
$reffile = "";  # -"-

while (@ARGV) {
    $arg = shift @ARGV;
    if ($arg eq "-n") {			# user-defined value for $nwords
	&usage() unless (@ARGV);
	$nwords = shift @ARGV;
	&usage() unless ($nwords =~ m/^[0-9]+$/);
    }
    elsif ($arg eq "-rand") {		# user-defined value for $randseed
	&usage() unless (@ARGV);
	$randseed = shift @ARGV;
	&usage() unless ($randseed =~ m/^[0-9]+$/);
    }
    elsif ($arg eq "-refwords") {
	&usage() unless (@ARGV);
	$reffile = shift @ARGV;
    }
    else {
	&usage();
    }
}

srand($randseed);

%usefulword = ();

# Read in the list of useful words (if file specified)

if ($reffile) {
    open(REFFILE, $reffile) ||
	die "Error ($me): Unable to open file \"$reffile\" for reading.\n";
    while ($line = <REFFILE>) {
	chomp $line;
	$line =~ s/[ \t].*$//;
	$usefulword{$line} = 1;
    }
    close REFFILE;
}

# Read in data

@data = ();
while ($line = <>) {
    chomp $line;
    $line =~ s/, */, /g;  # Make sure there is a space after the comma
    if ($reffile) {
	($word = $line) =~ s/[ \t].*$//;
	push @data, $line if ($usefulword{$word});
    }
    else {
	push @data, $line;
    }
}

%usefulword = ();	# Free memory

$nallwords = scalar(@data);
$nwords = $nallwords if ($nwords > $nallwords); 

# Sort data into random order

for ($i = $nallwords - 1; $i >= 0; $i--) {
    $r = rand($i+1);
    $tmp = $data[$i];
    $data[$i] = $data[$r];
    $data[$r] = $tmp;
}

# The first $nwords in the list sorted by random are the sampled words.
# First pick out all morphemes that occur in these words:

$nneeded = 0;	# Number of words needed as pairs to the selected words

foreach $i (0 .. $nwords - 1) {
    ($word, @morphemes) = split(" ", $data[$i]);
    foreach $morpheme (@morphemes) {
	$morpheme =~ s/,$//;
	# A word containing this morpheme is needed as a pair to this word
	push @{$needspair{$morpheme}}, $i;
	$nneeded++;
    }
}

# Then select word pairs

$i = 0;
@pairsfound = ();
@correspmorphemes = ();

while (($nneeded > 0) && ($i < $nallwords)) {
    ($word, @morphemes) = split(" ", $data[$i]);
    foreach $morpheme (@morphemes) {
	$morpheme =~ s/,$//;
	$pairneeded = $needspair{$morpheme};
	if (defined $pairneeded) {
	    if ($pairneeded->[0] != $i) { # The word itself cannot be its pair
		# A pair was found: update "found" and "needed" lists
		$j = shift @{$pairneeded};
		# This (ith) word is a pair of the jth word:
		push @{$pairsfound[$j]}, $i;
		# Also store the morpheme that they have in common:
		push @{$correspmorphemes[$j]}, $morpheme;
		$nneeded--;
		delete $needspair{$morpheme} unless (@{$needspair{$morpheme}});
	    }
	}
    }
    $i++;
}

# Output

foreach $i (0 .. $nwords - 1) {

    # Make a hash of all morphemes occurring in the current (ith) word.
    # The values in the hash are empty lists so far:
    # Also: Store which morpheme belongs to which alternative analysis
    #
    %mymorphemes = ();
    ($word, $analysis) = split(" ", $data[$i], 2);
    @morphemes = split(" ", $analysis);
    @alts = split(", ", $analysis);
    %altref = ();
    $altn = 0;
    foreach $alt (@alts) {
        @mlist = split(" ", $alt);
        %{$altref{$altn}} = ();
        foreach $morpheme (@mlist) {
            $altref{$altn}{$morpheme} = 1;
            @{$mymorphemes{$morpheme}} = ();
        }
        $altn++;
    }

    # Go through all the (jth) words that are pairs of the current (ith) word.
    # Update the %mymorphemes hash to tell which of the pair words (j)
    # are linked to through a particular morpheme (indicated by the hash key).
    #
    # Fill another hash %theirmorphemes with info for each pair word (j) of
    # all the morphemes it has in common with the current word (i).
    #
    %theirmorphemes = ();
    if (defined @{$pairsfound[$i]}) {	# This word does have some pairs
	foreach $j (@{$pairsfound[$i]}) { # For each pair...
	    # ... find the linking morpheme
	    $morpheme_i = shift @{$correspmorphemes[$i]};
	    ($word_j, @morphemes_j) = split(" ", $data[$j]);

	    # Store fast look-up: this morpheme links current (ith) word to
	    # word_j (there may be many instances of the same morpheme:
	    # that's why it's a list of words.)
	    push @{$mymorphemes{$morpheme_i}}, $word_j;

            $altn = 0;
	    foreach $morpheme_j (@morphemes_j) { # For each morpheme in pair...
                $morpheme_j =~ s/,$//;
		# ...indicate *all* the morphemes that it has in common
		# with the current word (there might be more than the
		# linking morpheme):
                $theirmorphemes{$word_j}{$morpheme_j} = 1 
                    if (defined $mymorphemes{$morpheme_j});
	    }
	}
    }

    # Generate output: print the word together with its pairs, and after each
    # pair word the common morpheme(s) are shown within brackets.
    # (Each unique morpheme type is only shown once, even though it may occur
    # several times, e.g., the same ending occurring in every alternative
    # analysis of the word. This is on purpose: here we operate with morpheme
    # types, not tokens.)
    #
    ($word, @morphemes) = split(" ", $data[$i]);
    $outstr = "$word\t";
    $altn = 0;
    foreach $morpheme (@morphemes) {
	$comma = 0;
	$comma = 1 if $morpheme =~ s/,$//;
	if (defined @{$mymorphemes{$morpheme}}) {
	    # Print the pair word and linking morphemes
	    $word_j = shift @{$mymorphemes{$morpheme}};
	    if (defined $word_j) {
                $outstr .= "$word_j [";
                $ok = 0;
                foreach $m (sort keys %{$theirmorphemes{$word_j}}) {
                    if (defined($altref{$altn}{$m})) {
                        $outstr .= "$m,";
                        $ok = 1;
                    }
                }
                if (!$ok) {
                    die "No linking morphemes for word '$word'\n";
                }
                chop $outstr;
                $outstr .= ']';
	    }
	    else {
		# There were no more occurrences of this morpheme
		$outstr .= "~ [$morpheme]";
	    }
	}
	else {
	    # No other word containing this morpheme was found. (Such a
	    # word may exist, but in that case the morpheme has already been
            # linked with a morpheme in some other third word.)
	    $outstr .= "~ [$morpheme]";
	}
	$outstr .= "," if ($comma);
        $altn += 1 if ($comma);
	$outstr .= " ";
    }
    $outstr =~ s/ $//;
    print "$outstr\n";
}

# End of main program.

sub usage {
    die "Usage: $me [-n <nwords>] [-rand <randseed>] " .
	"[-refwords <wordlist_file>] < inputdata\n";
}
