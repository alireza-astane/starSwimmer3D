#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-1.0285898660290547,-0.16660841219614414,-1.332634488935444>, 1 }        
    sphere {  m*<0.1892936145845443,0.2834766533172497,8.582689826414594>, 1 }
    sphere {  m*<5.38149731049681,0.05988114993352964,-4.541615617875198>, 1 }
    sphere {  m*<-2.685774691494984,2.1623606106382103,-2.243476386806241>, 1}
    sphere { m*<-2.4179874704571525,-2.725331331765687,-2.05393010164367>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.1892936145845443,0.2834766533172497,8.582689826414594>, <-1.0285898660290547,-0.16660841219614414,-1.332634488935444>, 0.5 }
    cylinder { m*<5.38149731049681,0.05988114993352964,-4.541615617875198>, <-1.0285898660290547,-0.16660841219614414,-1.332634488935444>, 0.5}
    cylinder { m*<-2.685774691494984,2.1623606106382103,-2.243476386806241>, <-1.0285898660290547,-0.16660841219614414,-1.332634488935444>, 0.5 }
    cylinder {  m*<-2.4179874704571525,-2.725331331765687,-2.05393010164367>, <-1.0285898660290547,-0.16660841219614414,-1.332634488935444>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-1.0285898660290547,-0.16660841219614414,-1.332634488935444>, 1 }        
    sphere {  m*<0.1892936145845443,0.2834766533172497,8.582689826414594>, 1 }
    sphere {  m*<5.38149731049681,0.05988114993352964,-4.541615617875198>, 1 }
    sphere {  m*<-2.685774691494984,2.1623606106382103,-2.243476386806241>, 1}
    sphere { m*<-2.4179874704571525,-2.725331331765687,-2.05393010164367>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.1892936145845443,0.2834766533172497,8.582689826414594>, <-1.0285898660290547,-0.16660841219614414,-1.332634488935444>, 0.5 }
    cylinder { m*<5.38149731049681,0.05988114993352964,-4.541615617875198>, <-1.0285898660290547,-0.16660841219614414,-1.332634488935444>, 0.5}
    cylinder { m*<-2.685774691494984,2.1623606106382103,-2.243476386806241>, <-1.0285898660290547,-0.16660841219614414,-1.332634488935444>, 0.5 }
    cylinder {  m*<-2.4179874704571525,-2.725331331765687,-2.05393010164367>, <-1.0285898660290547,-0.16660841219614414,-1.332634488935444>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    