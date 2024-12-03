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
    sphere { m*<-0.21355932427838045,-0.10939822723524689,-1.0060005935812069>, 1 }        
    sphere {  m*<0.3569026289245143,0.19560175469273028,6.073508362384121>, 1 }
    sphere {  m*<2.5211490697278767,-0.007364251848872669,-2.2352101190323874>, 1 }
    sphere {  m*<-1.8351746841712704,2.219075717183352,-1.9799463589971744>, 1}
    sphere { m*<-1.5673874631334386,-2.6686162252205454,-1.7904000738346018>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3569026289245143,0.19560175469273028,6.073508362384121>, <-0.21355932427838045,-0.10939822723524689,-1.0060005935812069>, 0.5 }
    cylinder { m*<2.5211490697278767,-0.007364251848872669,-2.2352101190323874>, <-0.21355932427838045,-0.10939822723524689,-1.0060005935812069>, 0.5}
    cylinder { m*<-1.8351746841712704,2.219075717183352,-1.9799463589971744>, <-0.21355932427838045,-0.10939822723524689,-1.0060005935812069>, 0.5 }
    cylinder {  m*<-1.5673874631334386,-2.6686162252205454,-1.7904000738346018>, <-0.21355932427838045,-0.10939822723524689,-1.0060005935812069>, 0.5}

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
    sphere { m*<-0.21355932427838045,-0.10939822723524689,-1.0060005935812069>, 1 }        
    sphere {  m*<0.3569026289245143,0.19560175469273028,6.073508362384121>, 1 }
    sphere {  m*<2.5211490697278767,-0.007364251848872669,-2.2352101190323874>, 1 }
    sphere {  m*<-1.8351746841712704,2.219075717183352,-1.9799463589971744>, 1}
    sphere { m*<-1.5673874631334386,-2.6686162252205454,-1.7904000738346018>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.3569026289245143,0.19560175469273028,6.073508362384121>, <-0.21355932427838045,-0.10939822723524689,-1.0060005935812069>, 0.5 }
    cylinder { m*<2.5211490697278767,-0.007364251848872669,-2.2352101190323874>, <-0.21355932427838045,-0.10939822723524689,-1.0060005935812069>, 0.5}
    cylinder { m*<-1.8351746841712704,2.219075717183352,-1.9799463589971744>, <-0.21355932427838045,-0.10939822723524689,-1.0060005935812069>, 0.5 }
    cylinder {  m*<-1.5673874631334386,-2.6686162252205454,-1.7904000738346018>, <-0.21355932427838045,-0.10939822723524689,-1.0060005935812069>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    