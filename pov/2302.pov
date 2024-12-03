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
    sphere { m*<1.0445790698745532,0.4000467557859501,0.48349192748227565>, 1 }        
    sphere {  m*<1.2885752550392846,0.4318572390610433,3.4733825444452364>, 1 }
    sphere {  m*<3.7818224441018202,0.4318572390610432,-0.7438996640453825>, 1 }
    sphere {  m*<-2.972238916448433,6.748522963265162,-1.891510293580316>, 1}
    sphere { m*<-3.794855740254104,-7.869572144272331,-2.3772199902864894>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2885752550392846,0.4318572390610433,3.4733825444452364>, <1.0445790698745532,0.4000467557859501,0.48349192748227565>, 0.5 }
    cylinder { m*<3.7818224441018202,0.4318572390610432,-0.7438996640453825>, <1.0445790698745532,0.4000467557859501,0.48349192748227565>, 0.5}
    cylinder { m*<-2.972238916448433,6.748522963265162,-1.891510293580316>, <1.0445790698745532,0.4000467557859501,0.48349192748227565>, 0.5 }
    cylinder {  m*<-3.794855740254104,-7.869572144272331,-2.3772199902864894>, <1.0445790698745532,0.4000467557859501,0.48349192748227565>, 0.5}

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
    sphere { m*<1.0445790698745532,0.4000467557859501,0.48349192748227565>, 1 }        
    sphere {  m*<1.2885752550392846,0.4318572390610433,3.4733825444452364>, 1 }
    sphere {  m*<3.7818224441018202,0.4318572390610432,-0.7438996640453825>, 1 }
    sphere {  m*<-2.972238916448433,6.748522963265162,-1.891510293580316>, 1}
    sphere { m*<-3.794855740254104,-7.869572144272331,-2.3772199902864894>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2885752550392846,0.4318572390610433,3.4733825444452364>, <1.0445790698745532,0.4000467557859501,0.48349192748227565>, 0.5 }
    cylinder { m*<3.7818224441018202,0.4318572390610432,-0.7438996640453825>, <1.0445790698745532,0.4000467557859501,0.48349192748227565>, 0.5}
    cylinder { m*<-2.972238916448433,6.748522963265162,-1.891510293580316>, <1.0445790698745532,0.4000467557859501,0.48349192748227565>, 0.5 }
    cylinder {  m*<-3.794855740254104,-7.869572144272331,-2.3772199902864894>, <1.0445790698745532,0.4000467557859501,0.48349192748227565>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    