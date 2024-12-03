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
    sphere { m*<-0.1948540957442635,-0.09939739486977055,-0.7738662116348607>, 1 }        
    sphere {  m*<0.28910370982145034,0.15935276563207654,5.232114729907819>, 1 }
    sphere {  m*<2.539854298261994,0.002636580516603676,-2.0030757370860415>, 1 }
    sphere {  m*<-1.8164694556371535,2.2290765495488287,-1.7478119770508282>, 1}
    sphere { m*<-1.5486822345993216,-2.6586153928550686,-1.5582656918882556>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.28910370982145034,0.15935276563207654,5.232114729907819>, <-0.1948540957442635,-0.09939739486977055,-0.7738662116348607>, 0.5 }
    cylinder { m*<2.539854298261994,0.002636580516603676,-2.0030757370860415>, <-0.1948540957442635,-0.09939739486977055,-0.7738662116348607>, 0.5}
    cylinder { m*<-1.8164694556371535,2.2290765495488287,-1.7478119770508282>, <-0.1948540957442635,-0.09939739486977055,-0.7738662116348607>, 0.5 }
    cylinder {  m*<-1.5486822345993216,-2.6586153928550686,-1.5582656918882556>, <-0.1948540957442635,-0.09939739486977055,-0.7738662116348607>, 0.5}

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
    sphere { m*<-0.1948540957442635,-0.09939739486977055,-0.7738662116348607>, 1 }        
    sphere {  m*<0.28910370982145034,0.15935276563207654,5.232114729907819>, 1 }
    sphere {  m*<2.539854298261994,0.002636580516603676,-2.0030757370860415>, 1 }
    sphere {  m*<-1.8164694556371535,2.2290765495488287,-1.7478119770508282>, 1}
    sphere { m*<-1.5486822345993216,-2.6586153928550686,-1.5582656918882556>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.28910370982145034,0.15935276563207654,5.232114729907819>, <-0.1948540957442635,-0.09939739486977055,-0.7738662116348607>, 0.5 }
    cylinder { m*<2.539854298261994,0.002636580516603676,-2.0030757370860415>, <-0.1948540957442635,-0.09939739486977055,-0.7738662116348607>, 0.5}
    cylinder { m*<-1.8164694556371535,2.2290765495488287,-1.7478119770508282>, <-0.1948540957442635,-0.09939739486977055,-0.7738662116348607>, 0.5 }
    cylinder {  m*<-1.5486822345993216,-2.6586153928550686,-1.5582656918882556>, <-0.1948540957442635,-0.09939739486977055,-0.7738662116348607>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    