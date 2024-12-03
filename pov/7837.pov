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
    sphere { m*<-0.4004913211239756,-0.3927321306629272,-0.43502655032180493>, 1 }        
    sphere {  m*<1.0186761730761855,0.5972067832169898,9.414263546713338>, 1 }
    sphere {  m*<8.386463371398982,0.3121145324247281,-5.156413882360589>, 1 }
    sphere {  m*<-6.509499822290012,6.835195906045365,-3.665606979178982>, 1}
    sphere { m*<-4.032992314738902,-8.303610020547973,-2.117192533542813>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0186761730761855,0.5972067832169898,9.414263546713338>, <-0.4004913211239756,-0.3927321306629272,-0.43502655032180493>, 0.5 }
    cylinder { m*<8.386463371398982,0.3121145324247281,-5.156413882360589>, <-0.4004913211239756,-0.3927321306629272,-0.43502655032180493>, 0.5}
    cylinder { m*<-6.509499822290012,6.835195906045365,-3.665606979178982>, <-0.4004913211239756,-0.3927321306629272,-0.43502655032180493>, 0.5 }
    cylinder {  m*<-4.032992314738902,-8.303610020547973,-2.117192533542813>, <-0.4004913211239756,-0.3927321306629272,-0.43502655032180493>, 0.5}

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
    sphere { m*<-0.4004913211239756,-0.3927321306629272,-0.43502655032180493>, 1 }        
    sphere {  m*<1.0186761730761855,0.5972067832169898,9.414263546713338>, 1 }
    sphere {  m*<8.386463371398982,0.3121145324247281,-5.156413882360589>, 1 }
    sphere {  m*<-6.509499822290012,6.835195906045365,-3.665606979178982>, 1}
    sphere { m*<-4.032992314738902,-8.303610020547973,-2.117192533542813>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0186761730761855,0.5972067832169898,9.414263546713338>, <-0.4004913211239756,-0.3927321306629272,-0.43502655032180493>, 0.5 }
    cylinder { m*<8.386463371398982,0.3121145324247281,-5.156413882360589>, <-0.4004913211239756,-0.3927321306629272,-0.43502655032180493>, 0.5}
    cylinder { m*<-6.509499822290012,6.835195906045365,-3.665606979178982>, <-0.4004913211239756,-0.3927321306629272,-0.43502655032180493>, 0.5 }
    cylinder {  m*<-4.032992314738902,-8.303610020547973,-2.117192533542813>, <-0.4004913211239756,-0.3927321306629272,-0.43502655032180493>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    