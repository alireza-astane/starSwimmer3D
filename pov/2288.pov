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
    sphere { m*<1.0555241097135144,0.3821221022928596,0.48996336846234556>, 1 }        
    sphere {  m*<1.2995444951349688,0.4123632138142681,3.4798683590579698>, 1 }
    sphere {  m*<3.792791684197504,0.412363213814268,-0.7374138494326483>, 1 }
    sphere {  m*<-3.006235757408511,6.813843710351821,-1.9116119127375233>, 1}
    sphere { m*<-3.790359313482416,-7.882397860571058,-2.3745611700363627>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2995444951349688,0.4123632138142681,3.4798683590579698>, <1.0555241097135144,0.3821221022928596,0.48996336846234556>, 0.5 }
    cylinder { m*<3.792791684197504,0.412363213814268,-0.7374138494326483>, <1.0555241097135144,0.3821221022928596,0.48996336846234556>, 0.5}
    cylinder { m*<-3.006235757408511,6.813843710351821,-1.9116119127375233>, <1.0555241097135144,0.3821221022928596,0.48996336846234556>, 0.5 }
    cylinder {  m*<-3.790359313482416,-7.882397860571058,-2.3745611700363627>, <1.0555241097135144,0.3821221022928596,0.48996336846234556>, 0.5}

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
    sphere { m*<1.0555241097135144,0.3821221022928596,0.48996336846234556>, 1 }        
    sphere {  m*<1.2995444951349688,0.4123632138142681,3.4798683590579698>, 1 }
    sphere {  m*<3.792791684197504,0.412363213814268,-0.7374138494326483>, 1 }
    sphere {  m*<-3.006235757408511,6.813843710351821,-1.9116119127375233>, 1}
    sphere { m*<-3.790359313482416,-7.882397860571058,-2.3745611700363627>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2995444951349688,0.4123632138142681,3.4798683590579698>, <1.0555241097135144,0.3821221022928596,0.48996336846234556>, 0.5 }
    cylinder { m*<3.792791684197504,0.412363213814268,-0.7374138494326483>, <1.0555241097135144,0.3821221022928596,0.48996336846234556>, 0.5}
    cylinder { m*<-3.006235757408511,6.813843710351821,-1.9116119127375233>, <1.0555241097135144,0.3821221022928596,0.48996336846234556>, 0.5 }
    cylinder {  m*<-3.790359313482416,-7.882397860571058,-2.3745611700363627>, <1.0555241097135144,0.3821221022928596,0.48996336846234556>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    