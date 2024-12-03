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
    sphere { m*<-2.3573915529398824e-18,-2.4984897727623255e-18,0.6274957952834385>, 1 }        
    sphere {  m*<-2.9143381779844163e-18,-5.676081638891058e-18,7.134495795283459>, 1 }
    sphere {  m*<9.428090415820634,-5.589206211952038e-19,-2.7058375380498956>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.7058375380498956>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.7058375380498956>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-2.9143381779844163e-18,-5.676081638891058e-18,7.134495795283459>, <-2.3573915529398824e-18,-2.4984897727623255e-18,0.6274957952834385>, 0.5 }
    cylinder { m*<9.428090415820634,-5.589206211952038e-19,-2.7058375380498956>, <-2.3573915529398824e-18,-2.4984897727623255e-18,0.6274957952834385>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.7058375380498956>, <-2.3573915529398824e-18,-2.4984897727623255e-18,0.6274957952834385>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.7058375380498956>, <-2.3573915529398824e-18,-2.4984897727623255e-18,0.6274957952834385>, 0.5}

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
    sphere { m*<-2.3573915529398824e-18,-2.4984897727623255e-18,0.6274957952834385>, 1 }        
    sphere {  m*<-2.9143381779844163e-18,-5.676081638891058e-18,7.134495795283459>, 1 }
    sphere {  m*<9.428090415820634,-5.589206211952038e-19,-2.7058375380498956>, 1 }
    sphere {  m*<-4.714045207910317,8.16496580927726,-2.7058375380498956>, 1}
    sphere { m*<-4.714045207910317,-8.16496580927726,-2.7058375380498956>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-2.9143381779844163e-18,-5.676081638891058e-18,7.134495795283459>, <-2.3573915529398824e-18,-2.4984897727623255e-18,0.6274957952834385>, 0.5 }
    cylinder { m*<9.428090415820634,-5.589206211952038e-19,-2.7058375380498956>, <-2.3573915529398824e-18,-2.4984897727623255e-18,0.6274957952834385>, 0.5}
    cylinder { m*<-4.714045207910317,8.16496580927726,-2.7058375380498956>, <-2.3573915529398824e-18,-2.4984897727623255e-18,0.6274957952834385>, 0.5 }
    cylinder {  m*<-4.714045207910317,-8.16496580927726,-2.7058375380498956>, <-2.3573915529398824e-18,-2.4984897727623255e-18,0.6274957952834385>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    