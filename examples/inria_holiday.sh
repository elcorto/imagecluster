#!/bin/sh

# select 25 images
#   ./this.sh jpg/100*
#
# select 274 images
#   ./this.sh jpg/10*

if ! [ -d jpg ]; then
    for name in jpg1 jpg2; do
        wget ftp://ftp.inrialpes.fr/pub/lear/douze/data/${name}.tar.gz
        tar -xzf ${name}.tar.gz
    done
fi

mkdir -p pics
rm -rf pics/*
for x in $@; do
    f=$(echo "$x" | sed -re 's|jpg/||')
    ln -s $(readlink -f jpg/$f) pics/$f
done

echo "#images: $(ls pics | wc -l)"
