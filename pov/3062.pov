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
    sphere { m*<0.46424312386636474,1.0854634695839087,0.1409290791874509>, 1 }        
    sphere {  m*<0.7049782286080564,1.2141735477642341,3.1284838503080006>, 1 }
    sphere {  m*<3.198951517872621,1.1874974449702833,-1.0882804462637337>, 1 }
    sphere {  m*<-1.1573722360265246,3.4139374140025094,-0.8330166862285199>, 1}
    sphere { m*<-3.8495046025721376,-7.069058144663248,-2.3584291313932786>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7049782286080564,1.2141735477642341,3.1284838503080006>, <0.46424312386636474,1.0854634695839087,0.1409290791874509>, 0.5 }
    cylinder { m*<3.198951517872621,1.1874974449702833,-1.0882804462637337>, <0.46424312386636474,1.0854634695839087,0.1409290791874509>, 0.5}
    cylinder { m*<-1.1573722360265246,3.4139374140025094,-0.8330166862285199>, <0.46424312386636474,1.0854634695839087,0.1409290791874509>, 0.5 }
    cylinder {  m*<-3.8495046025721376,-7.069058144663248,-2.3584291313932786>, <0.46424312386636474,1.0854634695839087,0.1409290791874509>, 0.5}

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
    sphere { m*<0.46424312386636474,1.0854634695839087,0.1409290791874509>, 1 }        
    sphere {  m*<0.7049782286080564,1.2141735477642341,3.1284838503080006>, 1 }
    sphere {  m*<3.198951517872621,1.1874974449702833,-1.0882804462637337>, 1 }
    sphere {  m*<-1.1573722360265246,3.4139374140025094,-0.8330166862285199>, 1}
    sphere { m*<-3.8495046025721376,-7.069058144663248,-2.3584291313932786>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7049782286080564,1.2141735477642341,3.1284838503080006>, <0.46424312386636474,1.0854634695839087,0.1409290791874509>, 0.5 }
    cylinder { m*<3.198951517872621,1.1874974449702833,-1.0882804462637337>, <0.46424312386636474,1.0854634695839087,0.1409290791874509>, 0.5}
    cylinder { m*<-1.1573722360265246,3.4139374140025094,-0.8330166862285199>, <0.46424312386636474,1.0854634695839087,0.1409290791874509>, 0.5 }
    cylinder {  m*<-3.8495046025721376,-7.069058144663248,-2.3584291313932786>, <0.46424312386636474,1.0854634695839087,0.1409290791874509>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    