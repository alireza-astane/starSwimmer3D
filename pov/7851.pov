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
    sphere { m*<-0.39318256711461147,-0.376815091521872,-0.43164195733865274>, 1 }        
    sphere {  m*<1.0259849270855494,0.6131238223580449,9.417648139696489>, 1 }
    sphere {  m*<8.393772125408347,0.32803157156578333,-5.153029289377436>, 1 }
    sphere {  m*<-6.5021910682806485,6.851112945186418,-3.662222386195828>, 1}
    sphere { m*<-4.0658352748479984,-8.375135574820941,-2.1324016994538244>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0259849270855494,0.6131238223580449,9.417648139696489>, <-0.39318256711461147,-0.376815091521872,-0.43164195733865274>, 0.5 }
    cylinder { m*<8.393772125408347,0.32803157156578333,-5.153029289377436>, <-0.39318256711461147,-0.376815091521872,-0.43164195733865274>, 0.5}
    cylinder { m*<-6.5021910682806485,6.851112945186418,-3.662222386195828>, <-0.39318256711461147,-0.376815091521872,-0.43164195733865274>, 0.5 }
    cylinder {  m*<-4.0658352748479984,-8.375135574820941,-2.1324016994538244>, <-0.39318256711461147,-0.376815091521872,-0.43164195733865274>, 0.5}

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
    sphere { m*<-0.39318256711461147,-0.376815091521872,-0.43164195733865274>, 1 }        
    sphere {  m*<1.0259849270855494,0.6131238223580449,9.417648139696489>, 1 }
    sphere {  m*<8.393772125408347,0.32803157156578333,-5.153029289377436>, 1 }
    sphere {  m*<-6.5021910682806485,6.851112945186418,-3.662222386195828>, 1}
    sphere { m*<-4.0658352748479984,-8.375135574820941,-2.1324016994538244>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0259849270855494,0.6131238223580449,9.417648139696489>, <-0.39318256711461147,-0.376815091521872,-0.43164195733865274>, 0.5 }
    cylinder { m*<8.393772125408347,0.32803157156578333,-5.153029289377436>, <-0.39318256711461147,-0.376815091521872,-0.43164195733865274>, 0.5}
    cylinder { m*<-6.5021910682806485,6.851112945186418,-3.662222386195828>, <-0.39318256711461147,-0.376815091521872,-0.43164195733865274>, 0.5 }
    cylinder {  m*<-4.0658352748479984,-8.375135574820941,-2.1324016994538244>, <-0.39318256711461147,-0.376815091521872,-0.43164195733865274>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    