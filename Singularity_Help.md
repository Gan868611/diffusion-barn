- Create the Singularity Image on Ubuntu 18.04 or 20.04 machines

# Follow this

https://docs.sylabs.io/guides/3.0/user-guide/installation.html#install-from-source

## Install these

libfuse-dev libfuse3-dev libglib-dev libtool

## Install Go

- Purge old go From https://go.dev/doc/install

```
rm -rf /usr/local/go
```

- GO 1.20

```
export VERSION=1.20 OS=linux ARCH=amd64 && \
    wget https://dl.google.com/go/go$VERSION.$OS-$ARCH.tar.gz && \
    sudo tar -C /usr/local -xzvf go$VERSION.$OS-$ARCH.tar.gz && \
    rm go$VERSION.$OS-$ARCH.tar.gz
```

- Start the session sourcing go

```
echo 'export GOPATH=${HOME}/go' >> ~/.bashrc && \
    echo 'export PATH=/usr/local/go/bin:${PATH}:${GOPATH}/bin' >> ~/.bashrc && \
    source ~/.bashrc
```

## Singularity 4.0.2

- Get the github repo

```
go get -d github.com/sylabs/singularity
```

- go to the version

```
export VERSION=v4.0.2 # or another tag or branch if you like && \
    cd $GOPATH/src/github.com/sylabs/singularity && \
    git fetch && \
    git checkout $VERSION # omit this command to install the latest bleeding edge code from master
```

## Make/Compile, do this inside the $GOPATH/ ... /singularity direction

```
   ./mconfig && \
    make -C ./builddir && \
    sudo make -C ./builddir install

    ./mconfig --prefix=/opt/singularity
```

## Check by typing in the terminal

```
singularity
```

##### NOTE

- Don't need the section "download from release"
- at the end of a long command, it will tell you to do smth, do that!
