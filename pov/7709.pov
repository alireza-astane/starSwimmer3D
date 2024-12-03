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
    sphere { m*<-0.46672315581720064,-0.5369721343414227,-0.4656976883402386>, 1 }        
    sphere {  m*<0.9524443383829616,0.45296677953849507,9.383592408694913>, 1 }
    sphere {  m*<8.320231536705766,0.16787452874623288,-5.1870850203790235>, 1 }
    sphere {  m*<-6.575731656983235,6.69095590236687,-3.6962781171974157>, 1}
    sphere { m*<-3.732122763206213,-7.6483748844411075,-1.977863590243176>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9524443383829616,0.45296677953849507,9.383592408694913>, <-0.46672315581720064,-0.5369721343414227,-0.4656976883402386>, 0.5 }
    cylinder { m*<8.320231536705766,0.16787452874623288,-5.1870850203790235>, <-0.46672315581720064,-0.5369721343414227,-0.4656976883402386>, 0.5}
    cylinder { m*<-6.575731656983235,6.69095590236687,-3.6962781171974157>, <-0.46672315581720064,-0.5369721343414227,-0.4656976883402386>, 0.5 }
    cylinder {  m*<-3.732122763206213,-7.6483748844411075,-1.977863590243176>, <-0.46672315581720064,-0.5369721343414227,-0.4656976883402386>, 0.5}

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
    sphere { m*<-0.46672315581720064,-0.5369721343414227,-0.4656976883402386>, 1 }        
    sphere {  m*<0.9524443383829616,0.45296677953849507,9.383592408694913>, 1 }
    sphere {  m*<8.320231536705766,0.16787452874623288,-5.1870850203790235>, 1 }
    sphere {  m*<-6.575731656983235,6.69095590236687,-3.6962781171974157>, 1}
    sphere { m*<-3.732122763206213,-7.6483748844411075,-1.977863590243176>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9524443383829616,0.45296677953849507,9.383592408694913>, <-0.46672315581720064,-0.5369721343414227,-0.4656976883402386>, 0.5 }
    cylinder { m*<8.320231536705766,0.16787452874623288,-5.1870850203790235>, <-0.46672315581720064,-0.5369721343414227,-0.4656976883402386>, 0.5}
    cylinder { m*<-6.575731656983235,6.69095590236687,-3.6962781171974157>, <-0.46672315581720064,-0.5369721343414227,-0.4656976883402386>, 0.5 }
    cylinder {  m*<-3.732122763206213,-7.6483748844411075,-1.977863590243176>, <-0.46672315581720064,-0.5369721343414227,-0.4656976883402386>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    