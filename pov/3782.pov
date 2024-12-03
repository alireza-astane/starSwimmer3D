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
    sphere { m*<-0.032695337746997066,0.14607255403713237,-0.14699392951666512>, 1 }        
    sphere {  m*<0.20803976699469456,0.2747826322174578,2.8405608416038857>, 1 }
    sphere {  m*<2.7020130562592657,0.24810652942350697,-1.3762034549678521>, 1 }
    sphere {  m*<-1.6543106976398887,2.4745464984557355,-1.1209396949326371>, 1}
    sphere { m*<-2.0720117909089906,-3.7089628235947085,-1.328561013271691>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.20803976699469456,0.2747826322174578,2.8405608416038857>, <-0.032695337746997066,0.14607255403713237,-0.14699392951666512>, 0.5 }
    cylinder { m*<2.7020130562592657,0.24810652942350697,-1.3762034549678521>, <-0.032695337746997066,0.14607255403713237,-0.14699392951666512>, 0.5}
    cylinder { m*<-1.6543106976398887,2.4745464984557355,-1.1209396949326371>, <-0.032695337746997066,0.14607255403713237,-0.14699392951666512>, 0.5 }
    cylinder {  m*<-2.0720117909089906,-3.7089628235947085,-1.328561013271691>, <-0.032695337746997066,0.14607255403713237,-0.14699392951666512>, 0.5}

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
    sphere { m*<-0.032695337746997066,0.14607255403713237,-0.14699392951666512>, 1 }        
    sphere {  m*<0.20803976699469456,0.2747826322174578,2.8405608416038857>, 1 }
    sphere {  m*<2.7020130562592657,0.24810652942350697,-1.3762034549678521>, 1 }
    sphere {  m*<-1.6543106976398887,2.4745464984557355,-1.1209396949326371>, 1}
    sphere { m*<-2.0720117909089906,-3.7089628235947085,-1.328561013271691>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.20803976699469456,0.2747826322174578,2.8405608416038857>, <-0.032695337746997066,0.14607255403713237,-0.14699392951666512>, 0.5 }
    cylinder { m*<2.7020130562592657,0.24810652942350697,-1.3762034549678521>, <-0.032695337746997066,0.14607255403713237,-0.14699392951666512>, 0.5}
    cylinder { m*<-1.6543106976398887,2.4745464984557355,-1.1209396949326371>, <-0.032695337746997066,0.14607255403713237,-0.14699392951666512>, 0.5 }
    cylinder {  m*<-2.0720117909089906,-3.7089628235947085,-1.328561013271691>, <-0.032695337746997066,0.14607255403713237,-0.14699392951666512>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    