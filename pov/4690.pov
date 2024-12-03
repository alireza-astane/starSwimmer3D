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
    sphere { m*<-0.22668748002688427,-0.11641725320807492,-1.168922758109477>, 1 }        
    sphere {  m*<0.40219285859332776,0.2198163743583279,6.635566155647774>, 1 }
    sphere {  m*<2.5080209139793728,-0.0143832778217007,-2.3981322835606576>, 1 }
    sphere {  m*<-1.848302839919774,2.212056691210524,-2.1428685235254443>, 1}
    sphere { m*<-1.5805156188819423,-2.6756352511933734,-1.9533222383628714>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.40219285859332776,0.2198163743583279,6.635566155647774>, <-0.22668748002688427,-0.11641725320807492,-1.168922758109477>, 0.5 }
    cylinder { m*<2.5080209139793728,-0.0143832778217007,-2.3981322835606576>, <-0.22668748002688427,-0.11641725320807492,-1.168922758109477>, 0.5}
    cylinder { m*<-1.848302839919774,2.212056691210524,-2.1428685235254443>, <-0.22668748002688427,-0.11641725320807492,-1.168922758109477>, 0.5 }
    cylinder {  m*<-1.5805156188819423,-2.6756352511933734,-1.9533222383628714>, <-0.22668748002688427,-0.11641725320807492,-1.168922758109477>, 0.5}

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
    sphere { m*<-0.22668748002688427,-0.11641725320807492,-1.168922758109477>, 1 }        
    sphere {  m*<0.40219285859332776,0.2198163743583279,6.635566155647774>, 1 }
    sphere {  m*<2.5080209139793728,-0.0143832778217007,-2.3981322835606576>, 1 }
    sphere {  m*<-1.848302839919774,2.212056691210524,-2.1428685235254443>, 1}
    sphere { m*<-1.5805156188819423,-2.6756352511933734,-1.9533222383628714>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.40219285859332776,0.2198163743583279,6.635566155647774>, <-0.22668748002688427,-0.11641725320807492,-1.168922758109477>, 0.5 }
    cylinder { m*<2.5080209139793728,-0.0143832778217007,-2.3981322835606576>, <-0.22668748002688427,-0.11641725320807492,-1.168922758109477>, 0.5}
    cylinder { m*<-1.848302839919774,2.212056691210524,-2.1428685235254443>, <-0.22668748002688427,-0.11641725320807492,-1.168922758109477>, 0.5 }
    cylinder {  m*<-1.5805156188819423,-2.6756352511933734,-1.9533222383628714>, <-0.22668748002688427,-0.11641725320807492,-1.168922758109477>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    