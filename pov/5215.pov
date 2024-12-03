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
    sphere { m*<-0.5236757217121384,-0.1481885528421803,-1.5704158964621884>, 1 }        
    sphere {  m*<0.4273071591538183,0.2884497044096682,8.374674060244686>, 1 }
    sphere {  m*<3.5614716355748297,0.0013844824592229577,-3.480235835669985>, 1 }
    sphere {  m*<-2.1582934328379397,2.180439039138856,-2.5220007561263174>, 1}
    sphere { m*<-1.8905062118001081,-2.7072529032650414,-2.3324544709637465>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4273071591538183,0.2884497044096682,8.374674060244686>, <-0.5236757217121384,-0.1481885528421803,-1.5704158964621884>, 0.5 }
    cylinder { m*<3.5614716355748297,0.0013844824592229577,-3.480235835669985>, <-0.5236757217121384,-0.1481885528421803,-1.5704158964621884>, 0.5}
    cylinder { m*<-2.1582934328379397,2.180439039138856,-2.5220007561263174>, <-0.5236757217121384,-0.1481885528421803,-1.5704158964621884>, 0.5 }
    cylinder {  m*<-1.8905062118001081,-2.7072529032650414,-2.3324544709637465>, <-0.5236757217121384,-0.1481885528421803,-1.5704158964621884>, 0.5}

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
    sphere { m*<-0.5236757217121384,-0.1481885528421803,-1.5704158964621884>, 1 }        
    sphere {  m*<0.4273071591538183,0.2884497044096682,8.374674060244686>, 1 }
    sphere {  m*<3.5614716355748297,0.0013844824592229577,-3.480235835669985>, 1 }
    sphere {  m*<-2.1582934328379397,2.180439039138856,-2.5220007561263174>, 1}
    sphere { m*<-1.8905062118001081,-2.7072529032650414,-2.3324544709637465>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4273071591538183,0.2884497044096682,8.374674060244686>, <-0.5236757217121384,-0.1481885528421803,-1.5704158964621884>, 0.5 }
    cylinder { m*<3.5614716355748297,0.0013844824592229577,-3.480235835669985>, <-0.5236757217121384,-0.1481885528421803,-1.5704158964621884>, 0.5}
    cylinder { m*<-2.1582934328379397,2.180439039138856,-2.5220007561263174>, <-0.5236757217121384,-0.1481885528421803,-1.5704158964621884>, 0.5 }
    cylinder {  m*<-1.8905062118001081,-2.7072529032650414,-2.3324544709637465>, <-0.5236757217121384,-0.1481885528421803,-1.5704158964621884>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    