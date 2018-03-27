drop table if exists search;
create table search (
    id integer primary key autoincrement,
    uname text not null,
    numOfTweets integer not null
);

drop table if exists tweets;
create table tweets (
    id integer primary key autoincrement,
    tweet text not null,
    NBSE integer not null,
    NBSTS integer not null,
    SVMSE integer not null,
    SVMSTS integer not null,
    CNNSE integer not null,
    CNNSTS integer not null,
    AVG real not null,
    search_id integer not null,
    foreign key (search_id) references search(id)
            on update cascade
            on delete cascade
);