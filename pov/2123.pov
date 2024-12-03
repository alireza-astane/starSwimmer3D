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
    sphere { m*<1.1838302447041709,0.1665614821290738,0.5658267835229253>, 1 }        
    sphere {  m*<1.4280392942875058,0.1790253228871131,3.5558440651929386>, 1 }
    sphere {  m*<3.9212864833500425,0.1790253228871131,-0.6614381432976795>, 1 }
    sphere {  m*<-3.4006850577698855,7.584640438346236,-2.144842059641493>, 1}
    sphere { m*<-3.7348158360591976,-8.03973150581195,-2.3417173319239275>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.4280392942875058,0.1790253228871131,3.5558440651929386>, <1.1838302447041709,0.1665614821290738,0.5658267835229253>, 0.5 }
    cylinder { m*<3.9212864833500425,0.1790253228871131,-0.6614381432976795>, <1.1838302447041709,0.1665614821290738,0.5658267835229253>, 0.5}
    cylinder { m*<-3.4006850577698855,7.584640438346236,-2.144842059641493>, <1.1838302447041709,0.1665614821290738,0.5658267835229253>, 0.5 }
    cylinder {  m*<-3.7348158360591976,-8.03973150581195,-2.3417173319239275>, <1.1838302447041709,0.1665614821290738,0.5658267835229253>, 0.5}

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
    sphere { m*<1.1838302447041709,0.1665614821290738,0.5658267835229253>, 1 }        
    sphere {  m*<1.4280392942875058,0.1790253228871131,3.5558440651929386>, 1 }
    sphere {  m*<3.9212864833500425,0.1790253228871131,-0.6614381432976795>, 1 }
    sphere {  m*<-3.4006850577698855,7.584640438346236,-2.144842059641493>, 1}
    sphere { m*<-3.7348158360591976,-8.03973150581195,-2.3417173319239275>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.4280392942875058,0.1790253228871131,3.5558440651929386>, <1.1838302447041709,0.1665614821290738,0.5658267835229253>, 0.5 }
    cylinder { m*<3.9212864833500425,0.1790253228871131,-0.6614381432976795>, <1.1838302447041709,0.1665614821290738,0.5658267835229253>, 0.5}
    cylinder { m*<-3.4006850577698855,7.584640438346236,-2.144842059641493>, <1.1838302447041709,0.1665614821290738,0.5658267835229253>, 0.5 }
    cylinder {  m*<-3.7348158360591976,-8.03973150581195,-2.3417173319239275>, <1.1838302447041709,0.1665614821290738,0.5658267835229253>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    