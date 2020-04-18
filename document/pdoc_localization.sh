#!/bin/bash

if [ "$TOMOTOPY_LANG" = "kr" ]; then
    sed -i -E "s/Parameters<\/h2>/파라미터<\/h2>/g" $@
    sed -i -E "s/Added in version:/추가된 버전:/g" $@
    sed -i -E "s/Instance variables<\/h3>/인스턴스 변수<\/h3>/g" $@
    sed -i -E "s/Methods<\/h3>/메소드<\/h3>/g" $@
    sed -i -E "s/Inherited members<\/h3>/상속받은 메소드 및 변수<\/h3>/g" $@
    sed -i -E "s/Ancestors<\/h3>/부모 클래스<\/h3>/g" $@
    sed -i -E "s/Super-module<\/h3>/상위 모듈<\/h3>/g" $@
    sed -i -E "s/Sub-modules<\/a>/하위 모듈<\/a>/g" $@
    sed -i -E "s/Global variables<\/a>/전역 변수<\/a>/g" $@
    sed -i -E "s/Classes<\/a>/클래스<\/a>/g" $@
fi
