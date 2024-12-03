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
    sphere { m*<0.292144296990156,0.7601353112061165,0.041216104933508496>, 1 }        
    sphere {  m*<0.5328794017318476,0.8888453893864421,3.0287708760540584>, 1 }
    sphere {  m*<3.026852690996412,0.862169286592491,-1.1879934205176745>, 1 }
    sphere {  m*<-1.3294710629027344,3.0886092556247187,-0.9327296604824605>, 1}
    sphere { m*<-3.2855721979574777,-6.0030247848030225,-2.0316902548827933>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5328794017318476,0.8888453893864421,3.0287708760540584>, <0.292144296990156,0.7601353112061165,0.041216104933508496>, 0.5 }
    cylinder { m*<3.026852690996412,0.862169286592491,-1.1879934205176745>, <0.292144296990156,0.7601353112061165,0.041216104933508496>, 0.5}
    cylinder { m*<-1.3294710629027344,3.0886092556247187,-0.9327296604824605>, <0.292144296990156,0.7601353112061165,0.041216104933508496>, 0.5 }
    cylinder {  m*<-3.2855721979574777,-6.0030247848030225,-2.0316902548827933>, <0.292144296990156,0.7601353112061165,0.041216104933508496>, 0.5}

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
    sphere { m*<0.292144296990156,0.7601353112061165,0.041216104933508496>, 1 }        
    sphere {  m*<0.5328794017318476,0.8888453893864421,3.0287708760540584>, 1 }
    sphere {  m*<3.026852690996412,0.862169286592491,-1.1879934205176745>, 1 }
    sphere {  m*<-1.3294710629027344,3.0886092556247187,-0.9327296604824605>, 1}
    sphere { m*<-3.2855721979574777,-6.0030247848030225,-2.0316902548827933>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5328794017318476,0.8888453893864421,3.0287708760540584>, <0.292144296990156,0.7601353112061165,0.041216104933508496>, 0.5 }
    cylinder { m*<3.026852690996412,0.862169286592491,-1.1879934205176745>, <0.292144296990156,0.7601353112061165,0.041216104933508496>, 0.5}
    cylinder { m*<-1.3294710629027344,3.0886092556247187,-0.9327296604824605>, <0.292144296990156,0.7601353112061165,0.041216104933508496>, 0.5 }
    cylinder {  m*<-3.2855721979574777,-6.0030247848030225,-2.0316902548827933>, <0.292144296990156,0.7601353112061165,0.041216104933508496>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    