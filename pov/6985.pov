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
    sphere { m*<-0.8178938527491953,-1.258647395270976,-0.6289470930130721>, 1 }        
    sphere {  m*<0.6051711361697356,-0.3664325640080774,9.229053422939332>, 1 }
    sphere {  m*<7.960522574169709,-0.45535284000243365,-5.350439867105999>, 1 }
    sphere {  m*<-6.909002305564659,5.899251461811291,-3.7457851438225194>, 1}
    sphere { m*<-2.032717907649951,-3.935714453789959,-1.2254927885244356>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6051711361697356,-0.3664325640080774,9.229053422939332>, <-0.8178938527491953,-1.258647395270976,-0.6289470930130721>, 0.5 }
    cylinder { m*<7.960522574169709,-0.45535284000243365,-5.350439867105999>, <-0.8178938527491953,-1.258647395270976,-0.6289470930130721>, 0.5}
    cylinder { m*<-6.909002305564659,5.899251461811291,-3.7457851438225194>, <-0.8178938527491953,-1.258647395270976,-0.6289470930130721>, 0.5 }
    cylinder {  m*<-2.032717907649951,-3.935714453789959,-1.2254927885244356>, <-0.8178938527491953,-1.258647395270976,-0.6289470930130721>, 0.5}

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
    sphere { m*<-0.8178938527491953,-1.258647395270976,-0.6289470930130721>, 1 }        
    sphere {  m*<0.6051711361697356,-0.3664325640080774,9.229053422939332>, 1 }
    sphere {  m*<7.960522574169709,-0.45535284000243365,-5.350439867105999>, 1 }
    sphere {  m*<-6.909002305564659,5.899251461811291,-3.7457851438225194>, 1}
    sphere { m*<-2.032717907649951,-3.935714453789959,-1.2254927885244356>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6051711361697356,-0.3664325640080774,9.229053422939332>, <-0.8178938527491953,-1.258647395270976,-0.6289470930130721>, 0.5 }
    cylinder { m*<7.960522574169709,-0.45535284000243365,-5.350439867105999>, <-0.8178938527491953,-1.258647395270976,-0.6289470930130721>, 0.5}
    cylinder { m*<-6.909002305564659,5.899251461811291,-3.7457851438225194>, <-0.8178938527491953,-1.258647395270976,-0.6289470930130721>, 0.5 }
    cylinder {  m*<-2.032717907649951,-3.935714453789959,-1.2254927885244356>, <-0.8178938527491953,-1.258647395270976,-0.6289470930130721>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    