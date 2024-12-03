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
    sphere { m*<0.8734657599550486,0.6702387941170991,0.3823186977591654>, 1 }        
    sphere {  m*<1.1168960159113193,0.7275874056849118,3.3718736566027054>, 1 }
    sphere {  m*<3.610143204973855,0.7275874056849115,-0.8454085518879126>, 1 }
    sphere {  m*<-2.430076109230767,5.733043552849029,-1.5709419056480647>, 1}
    sphere { m*<-3.8604581151936723,-7.68141162033792,-2.4160119308527177>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.1168960159113193,0.7275874056849118,3.3718736566027054>, <0.8734657599550486,0.6702387941170991,0.3823186977591654>, 0.5 }
    cylinder { m*<3.610143204973855,0.7275874056849115,-0.8454085518879126>, <0.8734657599550486,0.6702387941170991,0.3823186977591654>, 0.5}
    cylinder { m*<-2.430076109230767,5.733043552849029,-1.5709419056480647>, <0.8734657599550486,0.6702387941170991,0.3823186977591654>, 0.5 }
    cylinder {  m*<-3.8604581151936723,-7.68141162033792,-2.4160119308527177>, <0.8734657599550486,0.6702387941170991,0.3823186977591654>, 0.5}

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
    sphere { m*<0.8734657599550486,0.6702387941170991,0.3823186977591654>, 1 }        
    sphere {  m*<1.1168960159113193,0.7275874056849118,3.3718736566027054>, 1 }
    sphere {  m*<3.610143204973855,0.7275874056849115,-0.8454085518879126>, 1 }
    sphere {  m*<-2.430076109230767,5.733043552849029,-1.5709419056480647>, 1}
    sphere { m*<-3.8604581151936723,-7.68141162033792,-2.4160119308527177>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.1168960159113193,0.7275874056849118,3.3718736566027054>, <0.8734657599550486,0.6702387941170991,0.3823186977591654>, 0.5 }
    cylinder { m*<3.610143204973855,0.7275874056849115,-0.8454085518879126>, <0.8734657599550486,0.6702387941170991,0.3823186977591654>, 0.5}
    cylinder { m*<-2.430076109230767,5.733043552849029,-1.5709419056480647>, <0.8734657599550486,0.6702387941170991,0.3823186977591654>, 0.5 }
    cylinder {  m*<-3.8604581151936723,-7.68141162033792,-2.4160119308527177>, <0.8734657599550486,0.6702387941170991,0.3823186977591654>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    