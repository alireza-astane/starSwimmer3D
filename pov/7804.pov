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
    sphere { m*<-0.41767023969599854,-0.43014446099468273,-0.4429818936241679>, 1 }        
    sphere {  m*<1.0014972545041627,0.5597944528852346,9.406308203410978>, 1 }
    sphere {  m*<8.369284452826959,0.2747022020929726,-5.164369225662952>, 1 }
    sphere {  m*<-6.526678740862034,6.79778357571361,-3.6735623224813443>, 1}
    sphere { m*<-3.9555279071745573,-8.134907666403812,-2.0813197308803613>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0014972545041627,0.5597944528852346,9.406308203410978>, <-0.41767023969599854,-0.43014446099468273,-0.4429818936241679>, 0.5 }
    cylinder { m*<8.369284452826959,0.2747022020929726,-5.164369225662952>, <-0.41767023969599854,-0.43014446099468273,-0.4429818936241679>, 0.5}
    cylinder { m*<-6.526678740862034,6.79778357571361,-3.6735623224813443>, <-0.41767023969599854,-0.43014446099468273,-0.4429818936241679>, 0.5 }
    cylinder {  m*<-3.9555279071745573,-8.134907666403812,-2.0813197308803613>, <-0.41767023969599854,-0.43014446099468273,-0.4429818936241679>, 0.5}

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
    sphere { m*<-0.41767023969599854,-0.43014446099468273,-0.4429818936241679>, 1 }        
    sphere {  m*<1.0014972545041627,0.5597944528852346,9.406308203410978>, 1 }
    sphere {  m*<8.369284452826959,0.2747022020929726,-5.164369225662952>, 1 }
    sphere {  m*<-6.526678740862034,6.79778357571361,-3.6735623224813443>, 1}
    sphere { m*<-3.9555279071745573,-8.134907666403812,-2.0813197308803613>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0014972545041627,0.5597944528852346,9.406308203410978>, <-0.41767023969599854,-0.43014446099468273,-0.4429818936241679>, 0.5 }
    cylinder { m*<8.369284452826959,0.2747022020929726,-5.164369225662952>, <-0.41767023969599854,-0.43014446099468273,-0.4429818936241679>, 0.5}
    cylinder { m*<-6.526678740862034,6.79778357571361,-3.6735623224813443>, <-0.41767023969599854,-0.43014446099468273,-0.4429818936241679>, 0.5 }
    cylinder {  m*<-3.9555279071745573,-8.134907666403812,-2.0813197308803613>, <-0.41767023969599854,-0.43014446099468273,-0.4429818936241679>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    