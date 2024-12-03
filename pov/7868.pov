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
    sphere { m*<-0.3842913849468638,-0.3574518328333238,-0.42752456157757895>, 1 }        
    sphere {  m*<1.0348761092532972,0.6324870810465932,9.421765535457565>, 1 }
    sphere {  m*<8.402663307576095,0.3473948302543317,-5.148911893616361>, 1 }
    sphere {  m*<-6.4932998861129,6.870476203874966,-3.6581049904347536>, 1}
    sphere { m*<-4.10569974553838,-8.461952608135137,-2.1508624394928066>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0348761092532972,0.6324870810465932,9.421765535457565>, <-0.3842913849468638,-0.3574518328333238,-0.42752456157757895>, 0.5 }
    cylinder { m*<8.402663307576095,0.3473948302543317,-5.148911893616361>, <-0.3842913849468638,-0.3574518328333238,-0.42752456157757895>, 0.5}
    cylinder { m*<-6.4932998861129,6.870476203874966,-3.6581049904347536>, <-0.3842913849468638,-0.3574518328333238,-0.42752456157757895>, 0.5 }
    cylinder {  m*<-4.10569974553838,-8.461952608135137,-2.1508624394928066>, <-0.3842913849468638,-0.3574518328333238,-0.42752456157757895>, 0.5}

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
    sphere { m*<-0.3842913849468638,-0.3574518328333238,-0.42752456157757895>, 1 }        
    sphere {  m*<1.0348761092532972,0.6324870810465932,9.421765535457565>, 1 }
    sphere {  m*<8.402663307576095,0.3473948302543317,-5.148911893616361>, 1 }
    sphere {  m*<-6.4932998861129,6.870476203874966,-3.6581049904347536>, 1}
    sphere { m*<-4.10569974553838,-8.461952608135137,-2.1508624394928066>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0348761092532972,0.6324870810465932,9.421765535457565>, <-0.3842913849468638,-0.3574518328333238,-0.42752456157757895>, 0.5 }
    cylinder { m*<8.402663307576095,0.3473948302543317,-5.148911893616361>, <-0.3842913849468638,-0.3574518328333238,-0.42752456157757895>, 0.5}
    cylinder { m*<-6.4932998861129,6.870476203874966,-3.6581049904347536>, <-0.3842913849468638,-0.3574518328333238,-0.42752456157757895>, 0.5 }
    cylinder {  m*<-4.10569974553838,-8.461952608135137,-2.1508624394928066>, <-0.3842913849468638,-0.3574518328333238,-0.42752456157757895>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    