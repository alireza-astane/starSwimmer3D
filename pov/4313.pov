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
    sphere { m*<-0.1811566486201937,-0.09207399476947531,-0.6038790651752899>, 1 }        
    sphere {  m*<0.23595704286224395,0.1309376673576349,4.572557501586253>, 1 }
    sphere {  m*<2.553551745386063,0.009959980616898845,-1.8330885906264722>, 1 }
    sphere {  m*<-1.8027720085130838,2.2363999496491234,-1.577824830591259>, 1}
    sphere { m*<-1.534984787475252,-2.651291992754774,-1.3882785454286863>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.23595704286224395,0.1309376673576349,4.572557501586253>, <-0.1811566486201937,-0.09207399476947531,-0.6038790651752899>, 0.5 }
    cylinder { m*<2.553551745386063,0.009959980616898845,-1.8330885906264722>, <-0.1811566486201937,-0.09207399476947531,-0.6038790651752899>, 0.5}
    cylinder { m*<-1.8027720085130838,2.2363999496491234,-1.577824830591259>, <-0.1811566486201937,-0.09207399476947531,-0.6038790651752899>, 0.5 }
    cylinder {  m*<-1.534984787475252,-2.651291992754774,-1.3882785454286863>, <-0.1811566486201937,-0.09207399476947531,-0.6038790651752899>, 0.5}

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
    sphere { m*<-0.1811566486201937,-0.09207399476947531,-0.6038790651752899>, 1 }        
    sphere {  m*<0.23595704286224395,0.1309376673576349,4.572557501586253>, 1 }
    sphere {  m*<2.553551745386063,0.009959980616898845,-1.8330885906264722>, 1 }
    sphere {  m*<-1.8027720085130838,2.2363999496491234,-1.577824830591259>, 1}
    sphere { m*<-1.534984787475252,-2.651291992754774,-1.3882785454286863>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.23595704286224395,0.1309376673576349,4.572557501586253>, <-0.1811566486201937,-0.09207399476947531,-0.6038790651752899>, 0.5 }
    cylinder { m*<2.553551745386063,0.009959980616898845,-1.8330885906264722>, <-0.1811566486201937,-0.09207399476947531,-0.6038790651752899>, 0.5}
    cylinder { m*<-1.8027720085130838,2.2363999496491234,-1.577824830591259>, <-0.1811566486201937,-0.09207399476947531,-0.6038790651752899>, 0.5 }
    cylinder {  m*<-1.534984787475252,-2.651291992754774,-1.3882785454286863>, <-0.1811566486201937,-0.09207399476947531,-0.6038790651752899>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    