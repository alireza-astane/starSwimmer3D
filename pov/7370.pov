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
    sphere { m*<-0.6359363593926494,-0.9054854515282916,-0.5440582161052165>, 1 }        
    sphere {  m*<0.7832311348075128,0.08445346235162599,9.305231880929934>, 1 }
    sphere {  m*<8.15101833313031,-0.20063878844063643,-5.2654455481439975>, 1 }
    sphere {  m*<-6.744944860558677,6.32244258518001,-3.7746386449623923>, 1}
    sphere { m*<-2.929090889198963,-5.899528261102686,-1.6059895276423226>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7832311348075128,0.08445346235162599,9.305231880929934>, <-0.6359363593926494,-0.9054854515282916,-0.5440582161052165>, 0.5 }
    cylinder { m*<8.15101833313031,-0.20063878844063643,-5.2654455481439975>, <-0.6359363593926494,-0.9054854515282916,-0.5440582161052165>, 0.5}
    cylinder { m*<-6.744944860558677,6.32244258518001,-3.7746386449623923>, <-0.6359363593926494,-0.9054854515282916,-0.5440582161052165>, 0.5 }
    cylinder {  m*<-2.929090889198963,-5.899528261102686,-1.6059895276423226>, <-0.6359363593926494,-0.9054854515282916,-0.5440582161052165>, 0.5}

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
    sphere { m*<-0.6359363593926494,-0.9054854515282916,-0.5440582161052165>, 1 }        
    sphere {  m*<0.7832311348075128,0.08445346235162599,9.305231880929934>, 1 }
    sphere {  m*<8.15101833313031,-0.20063878844063643,-5.2654455481439975>, 1 }
    sphere {  m*<-6.744944860558677,6.32244258518001,-3.7746386449623923>, 1}
    sphere { m*<-2.929090889198963,-5.899528261102686,-1.6059895276423226>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7832311348075128,0.08445346235162599,9.305231880929934>, <-0.6359363593926494,-0.9054854515282916,-0.5440582161052165>, 0.5 }
    cylinder { m*<8.15101833313031,-0.20063878844063643,-5.2654455481439975>, <-0.6359363593926494,-0.9054854515282916,-0.5440582161052165>, 0.5}
    cylinder { m*<-6.744944860558677,6.32244258518001,-3.7746386449623923>, <-0.6359363593926494,-0.9054854515282916,-0.5440582161052165>, 0.5 }
    cylinder {  m*<-2.929090889198963,-5.899528261102686,-1.6059895276423226>, <-0.6359363593926494,-0.9054854515282916,-0.5440582161052165>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    