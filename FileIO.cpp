#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int FileRow( char *filename )
{
    FILE *fp = fopen(filename,"r");
    int row = 0;
    char check = 0;
    char *buffer = (char*)malloc(sizeof(char)*65535);

    if( fp == NULL )
        return 0;

    while( check != EOF )
    {
        fscanf( fp, "%[^\n]", buffer);  //讀取一行字串，直到換行
        if( strlen(buffer) != 0 )     //檢查讀取字串長度，有字的話 row+1
            row++;
        strcpy(buffer,"");
        check = fgetc(fp);
    }

    free(buffer);
    fclose(fp);
    return row;
}
