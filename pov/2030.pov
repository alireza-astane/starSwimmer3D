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
    sphere { m*<1.2555128775725326,0.04196743497546594,0.6082106645978147>, 1 }        
    sphere {  m*<1.499758245976026,0.04501085877945266,3.5982498170365718>, 1 }
    sphere {  m*<3.993005435038563,0.04501085877945267,-0.6190323914540445>, 1 }
    sphere {  m*<-3.618517949061357,8.019800855296719,-2.273642784669307>, 1}
    sphere { m*<-3.701478942469724,-8.13312166357671,-2.3220046827014>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.499758245976026,0.04501085877945266,3.5982498170365718>, <1.2555128775725326,0.04196743497546594,0.6082106645978147>, 0.5 }
    cylinder { m*<3.993005435038563,0.04501085877945267,-0.6190323914540445>, <1.2555128775725326,0.04196743497546594,0.6082106645978147>, 0.5}
    cylinder { m*<-3.618517949061357,8.019800855296719,-2.273642784669307>, <1.2555128775725326,0.04196743497546594,0.6082106645978147>, 0.5 }
    cylinder {  m*<-3.701478942469724,-8.13312166357671,-2.3220046827014>, <1.2555128775725326,0.04196743497546594,0.6082106645978147>, 0.5}

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
    sphere { m*<1.2555128775725326,0.04196743497546594,0.6082106645978147>, 1 }        
    sphere {  m*<1.499758245976026,0.04501085877945266,3.5982498170365718>, 1 }
    sphere {  m*<3.993005435038563,0.04501085877945267,-0.6190323914540445>, 1 }
    sphere {  m*<-3.618517949061357,8.019800855296719,-2.273642784669307>, 1}
    sphere { m*<-3.701478942469724,-8.13312166357671,-2.3220046827014>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.499758245976026,0.04501085877945266,3.5982498170365718>, <1.2555128775725326,0.04196743497546594,0.6082106645978147>, 0.5 }
    cylinder { m*<3.993005435038563,0.04501085877945267,-0.6190323914540445>, <1.2555128775725326,0.04196743497546594,0.6082106645978147>, 0.5}
    cylinder { m*<-3.618517949061357,8.019800855296719,-2.273642784669307>, <1.2555128775725326,0.04196743497546594,0.6082106645978147>, 0.5 }
    cylinder {  m*<-3.701478942469724,-8.13312166357671,-2.3220046827014>, <1.2555128775725326,0.04196743497546594,0.6082106645978147>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    