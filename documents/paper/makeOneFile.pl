#!/usr/bin/perl -w

#
# Make one single tex file by resolving all \input directive.
#
# usage:
#  ./makeOneFile.pl main.tex
#

sub main {

  if(@ARGV < 1){
    die("Not enough arguments.\n "
      ."Please enter the name of the main tex file as argument");
  }

  open(INPUTFILE, "$ARGV[0]") 
    or die ("Cannot open file '$ARGV[0]' for reading\n");

  my $newTexFileName = $ARGV[0];

  if($newTexFileName =~ m/_linenum/){
    die("File has already been processed.\n "
      ."Please enter the name of the main tex file as argument");
  }

  $newTexFileName =~ s/.tex//;
  $newTexFileName = $newTexFileName."_single.tex";

  open(OUTPUTFILE, "> $newTexFileName");  

  print "Creating new Tex file $newTexFileName \n";

  my $mainData;
  while(<INPUTFILE>) {
    $mainData .= $_;
  }

  my ($input,$input_name);

  while(($mainData =~ m/\n\s*\\input\s*{\s*([\w\-]+[\/\w\-]*)\.tex\s*}/)
    ||($mainData =~ m/\n\s*\\input\s*{\s*([\w\-]+[\/\w\-]*)\s*}/)
    ||($mainData =~ m/\n\s*\\input\s+([\w\-]+[\/\w\-]*)\.tex/)
    ||($mainData =~ m/\n\s*\\input\s+([\w\-]+[\/\w\-]*)/)
    ||($mainData =~ m/\n\s*\\include\s*{\s*([\w\-]+[\/\w\-]*)\.tex\s*}/)
    ||($mainData =~ m/\n\s*\\include\s*{\s*([\w\-]+[\/\w\-]*)\s*}/)
    ||($mainData =~ m/\n\s*\\include\s+([\w\-]+[\/\w\-]*)\.tex/)
    ||($mainData =~ m/\n\s*\\include\s+([\w\-]+[\/\w\-]*)/)

  )
  {

    $input_name = $1;
    print "opening '$input_name.tex' \n";
    open(NEWINPUTFILE, "$input_name.tex") or die ("Cannot open file '$input_name.tex' for reading\n");
    while(<NEWINPUTFILE>) {
      $input .= $_;
    }
    $mainData =~ s/\n(\s*)\\input{\s*$input_name\.tex\s*}/\n$1$input/g; 
    $mainData =~ s/\n(\s*)\\input{\s*$input_name\s*}/\n$1$input/g; 
    $mainData =~ s/\n(\s*)\\input\s+$input_name\.tex/\n$1$input/g; 
    $mainData =~ s/\n(\s*)\\input\s+$input_name/\n$1$input/g; 
    $mainData =~ s/\n(\s*)\\include{\s*$input_name\.tex\s*}/\n$1$input/g; 
    $mainData =~ s/\n(\s*)\\include{\s*$input_name\s*}/\n$1$input/g; 
    $mainData =~ s/\n(\s*)\\include\s+$input_name\.tex/\n$1$input/g; 
    $mainData =~ s/\n(\s*)\\include\s+$input_name/\n$1$input/g; 

    close NEWINPUTFILE;
    $input = "";
    $input_name = "";
  }

  print OUTPUTFILE $mainData;
  close OUTPUTFILE;
}

&main
