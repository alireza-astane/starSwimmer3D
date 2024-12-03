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
    sphere { m*<0.42146831472493024,1.0046038255320981,0.11614562470344603>, 1 }        
    sphere {  m*<0.662203419466622,1.1333139037124236,3.103700395823997>, 1 }
    sphere {  m*<3.156176708731187,1.1066378009184725,-1.113063900747738>, 1 }
    sphere {  m*<-1.2001470451679594,3.333077769950699,-0.8578001407125243>, 1}
    sphere { m*<-3.712220269245849,-6.809541794983013,-2.2788874550035834>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.662203419466622,1.1333139037124236,3.103700395823997>, <0.42146831472493024,1.0046038255320981,0.11614562470344603>, 0.5 }
    cylinder { m*<3.156176708731187,1.1066378009184725,-1.113063900747738>, <0.42146831472493024,1.0046038255320981,0.11614562470344603>, 0.5}
    cylinder { m*<-1.2001470451679594,3.333077769950699,-0.8578001407125243>, <0.42146831472493024,1.0046038255320981,0.11614562470344603>, 0.5 }
    cylinder {  m*<-3.712220269245849,-6.809541794983013,-2.2788874550035834>, <0.42146831472493024,1.0046038255320981,0.11614562470344603>, 0.5}

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
    sphere { m*<0.42146831472493024,1.0046038255320981,0.11614562470344603>, 1 }        
    sphere {  m*<0.662203419466622,1.1333139037124236,3.103700395823997>, 1 }
    sphere {  m*<3.156176708731187,1.1066378009184725,-1.113063900747738>, 1 }
    sphere {  m*<-1.2001470451679594,3.333077769950699,-0.8578001407125243>, 1}
    sphere { m*<-3.712220269245849,-6.809541794983013,-2.2788874550035834>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.662203419466622,1.1333139037124236,3.103700395823997>, <0.42146831472493024,1.0046038255320981,0.11614562470344603>, 0.5 }
    cylinder { m*<3.156176708731187,1.1066378009184725,-1.113063900747738>, <0.42146831472493024,1.0046038255320981,0.11614562470344603>, 0.5}
    cylinder { m*<-1.2001470451679594,3.333077769950699,-0.8578001407125243>, <0.42146831472493024,1.0046038255320981,0.11614562470344603>, 0.5 }
    cylinder {  m*<-3.712220269245849,-6.809541794983013,-2.2788874550035834>, <0.42146831472493024,1.0046038255320981,0.11614562470344603>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    